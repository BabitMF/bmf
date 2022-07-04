/*
 * Copyright 2023 Babit Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "../include/common.h"
#include "../include/graph.h"
#include "../include/module_factory.h"
#include "../include/running_info_collector.h"

#include <bmf/sdk/log.h>
#include <bmf/sdk/trace.h>

#include <csignal>
#include <iomanip>
#include <iostream>

BEGIN_BMF_ENGINE_NS
    USE_BMF_SDK_NS
    std::vector<Graph *> g_ptr;

    void terminate(int signum) {
        std::cout << "terminated, ending bmf gracefully..." << std::endl;
        for (auto p:g_ptr)
            p->quit_gracefully();
    }

    void interrupted(int signum) {
        std::cout << "interrupted, ending bmf gracefully..." << std::endl;
        for (auto p:g_ptr)
            p->quit_gracefully();
    }

    Graph::Graph(GraphConfig graph_config, std::map<int, std::shared_ptr<Module> > pre_modules,
                 std::map<int, std::shared_ptr<ModuleCallbackLayer> > callback_bindings) {
        std::signal(SIGTERM, terminate);
        std::signal(SIGINT, interrupted);
        BMFLOG_CONFIGURE();
        BMFLOG(BMF_INFO) << "BMF Version: " << BMF_VERSION;
        BMFLOG(BMF_INFO) << "BMF Commit: " << BMF_COMMIT;
        BMFLOG(BMF_INFO) << "start init graph";
        BMF_TRACE(GRAPH_START, "Init");
        init(graph_config, pre_modules, callback_bindings);
        g_ptr.push_back(this);
    }

    void Graph::init(GraphConfig graph_config, std::map<int, std::shared_ptr<Module> > &pre_modules,
                     std::map<int, std::shared_ptr<ModuleCallbackLayer> > &callback_bindings) {
        graph_config_ = graph_config;
        pre_modules_ = pre_modules;
        callback_bindings_ = callback_bindings;
        mode_ = graph_config.get_mode();

        scheduler_count_ = 2;
        if (graph_config.get_option().json_value_.count("scheduler_count"))
            scheduler_count_ = graph_config.get_option().json_value_.at("scheduler_count").get<int>();

        SchedulerCallBack scheduler_callback;
        scheduler_callback.get_node_ = [this](int node_id, std::shared_ptr<Node> &node) -> int {
            return this->get_node(node_id, node);
        };
        scheduler_callback.close_report_ = [this](int node_id) -> int {
            std::lock_guard<std::mutex> _(this->con_var_mutex_);
            this->closed_count_++;
            if (this->closed_count_ == this->nodes_.size())
                this->cond_close_.notify_one();
            //if (this->all_nodes_done())
            //    this->cond_close_.notify_one();
            return 0;
        };
        scheduler_ = std::make_shared<Scheduler>(scheduler_callback, scheduler_count_);
        BMFLOG(BMF_INFO) << "scheduler count" << scheduler_count_;

        // create all nodes and output streams
        init_nodes();
        // retrieve all hungry check functions for all sources
        get_hungry_check_func_for_sources();

        // Init all graph input stream
        // graph input stream contains an output stream manager
        // which connect downstream input stream manager
        init_input_streams();

        // input streams that are not connected
        find_orphan_input_streams();

        for (auto &node:source_nodes_)
            scheduler_->add_or_remove_node(node->get_id(), true);
    }

    int
    Graph::get_hungry_check_func(std::shared_ptr<Node> &root_node, int output_idx, std::shared_ptr<Node> &curr_node) {
        std::map<int, std::shared_ptr<OutputStream>> output_streams;
        curr_node->get_output_streams(output_streams);
        for (auto &output_stream:output_streams) {
            if (curr_node == root_node && output_stream.first != output_idx) {
                continue;
            }
            for (auto &mirror_stream :output_stream.second->mirror_streams_) {
                int node_id = (mirror_stream.input_stream_manager_->node_id_);
                std::shared_ptr<Node> node;
                get_node(node_id, node);
                int stream_id = mirror_stream.stream_id_;
                if (node != nullptr) {
                    std::vector<std::function<bool()> > funcs;
                    node->get_hungry_check_func(stream_id, funcs);
                    for (auto func:funcs)
                        root_node->register_hungry_check_func(output_idx, func);
                    get_hungry_check_func(root_node, output_idx, node);
                }

            }
        }
        return 0;
    }

    int Graph::get_hungry_check_func_for_sources() {
        for (auto node:source_nodes_) {
            get_hungry_check_func(node, 0, node);
            get_hungry_check_func(node, 1, node);
        }
        return 0;
    }

    int Graph::init_nodes() {
        NodeCallBack callback;
        callback.get_node = [this](int node_id, std::shared_ptr<Node> &node) -> int {
            return this->get_node(node_id, node);
        };
        callback.throttled_cb = [this](int node_id, bool is_add) -> int {
            return this->scheduler_->add_or_remove_node(node_id, is_add);
        };
        callback.sched_required = [this](int node_id, bool is_add) -> int {
            return this->scheduler_->sched_required(node_id, is_add);
        };
        callback.scheduler_cb = [this](Task &task) -> int {
            return this->scheduler_->schedule_node(task);
        };
        callback.clear_cb = [this](int node_id, int scheduler_queue_id) -> int {
            return this->scheduler_->clear_task(node_id, scheduler_queue_id);
        };
        // init node
        for (auto &node_config:graph_config_.get_nodes()) {
            std::shared_ptr<Module> module_pre_allocated;
            auto node_id = node_config.get_id();
            if (pre_modules_.count(node_id) > 0)
                module_pre_allocated = pre_modules_[node_id];
            std::shared_ptr<Node> node;

            if (!callback_bindings_.count(node_id))
                callback_bindings_[node_id] = std::make_shared<ModuleCallbackLayer>();
            node = std::make_shared<Node>(node_id, node_config, callback, module_pre_allocated, mode_,
                                          callback_bindings_[node_id]);

            if (node_config.get_scheduler() < scheduler_count_) {
                node->set_scheduler_queue_id((node_config.get_scheduler()));
                BMFLOG(BMF_INFO) << "node:" << node->get_type() << " " << node->get_id() << " scheduler "
                                 << node_config.get_scheduler();
            } else {
                node->set_scheduler_queue_id(0);
                BMFLOG(BMF_WARNING) << "Node[" << node->get_id() << "](" << node->get_type()
                                    << ") scheduler exceed limit, will set to 0.";
                BMFLOG(BMF_INFO) << "node:" << node->get_type() << " " << node->get_id() << " scheduler " << 0;
            }
            nodes_[node_config.get_id()] = node;

            if ((node_config.get_input_streams().size()) == 0) {
                // if no input stream, it's a source node
                source_nodes_.push_back(node);
            }
        }

        // create connections
        for (auto &node_iter:nodes_) {
            std::map<int, std::shared_ptr<OutputStream>> output_streams;
            node_iter.second->get_output_streams(output_streams);
            // find all downstream connections for every output stream
            for (auto &output_stream:output_streams) {
                add_all_mirrors_for_output_stream(output_stream.second);
            }
        }
        for (auto &node_iter:nodes_) {
            std::map<int, std::shared_ptr<OutputStream>> output_streams;
            node_iter.second->get_output_streams(output_streams);
            for (auto &output_stream:output_streams)
                output_stream.second->add_upstream_nodes(node_iter.first);
        }
        // create graph output streams
        for (auto &graph_output_stream:graph_config_.output_streams) {
            for (auto &node:graph_config_.nodes) {
                int idx = 0;
                for (auto &out_s:node.output_streams) {
                    if (out_s.get_identifier() == graph_output_stream.get_identifier()) {
                        std::shared_ptr<InputStreamManager> input_manager;
                        create_input_stream_manager("immediate", -1, {out_s}, {}, InputStreamManagerCallBack(), 5,
                                                    input_manager);
                        auto g_out_s = std::make_shared<GraphOutputStream>();
                        g_out_s->set_manager(input_manager);

                        std::map<int, std::shared_ptr<OutputStream>> output_streams;
                        nodes_[node.id]->get_output_streams(output_streams);
                        output_streams[idx]->add_mirror_stream(input_manager, 0);

                        output_streams_[graph_output_stream.get_identifier()] = g_out_s;

                    }
                    ++idx;
                }
            }
        }

        return 0;
    }

    int Graph::init_input_streams() {
        // init all graph input streams
        for (auto &stream:graph_config_.get_input_streams()) {
            // create graph input stream
            std::shared_ptr<GraphInputStream> graph_input_stream = std::make_shared<GraphInputStream>();
            // create output stream manager and set it into input stream
            std::vector<StreamConfig> ss = {stream};
            std::shared_ptr<OutputStreamManager> manager = std::make_shared<OutputStreamManager>(ss);
            graph_input_stream->set_manager(manager);

            input_streams_[stream.get_identifier()] = graph_input_stream;

            std::shared_ptr<OutputStream> output_stream;
            manager->get_stream(0, output_stream);
            // link downstream nodes
            add_all_mirrors_for_output_stream(output_stream);
        }
        return 0;
    }

    int Graph::add_all_mirrors_for_output_stream(std::shared_ptr<OutputStream> &output_stream) {
        // go through all the nodes, find the input stream that
        // connected with graph input stream, add it to mirrors
        for (auto &node_iter :nodes_) {
            if (not node_iter.second->is_source()) {
                std::shared_ptr<InputStreamManager> input_stream_manager;
                node_iter.second->get_input_stream_manager(input_stream_manager);
                for (auto &input_stream:input_stream_manager->input_streams_) {
                    if (output_stream->identifier_ == input_stream.second->identifier_) {
                        output_stream->add_mirror_stream(input_stream_manager, input_stream.first);
                        input_stream.second->set_connected(true);
                    }
                }
            }
        }
        return 0;
    }

    int Graph::find_orphan_input_streams() {
        for (auto &node:nodes_) {
            std::map<int, std::shared_ptr<InputStream> > input_streams;
            node.second->get_input_streams(input_streams);
            for (auto &input_stream:input_streams) {
                if (not input_stream.second->is_connected()) {
                    orphan_streams_.push_back(input_stream.second);
                }
            }
        }
        return 0;
    }

    int Graph::start() {
        // start scheduler and it will start to schedule source nodes
        scheduler_->start();

        //TODO push eof to the orphan streams
        for (auto &stream:orphan_streams_) {
            std::shared_ptr<SafeQueue<Packet> > q = std::make_shared<SafeQueue<Packet> >();
            q->push(Packet::generate_eof_packet());
            stream->add_packets(q);
            BMFLOG(BMF_INFO) << "push eof to orphan stream " << stream->get_identifier();
        }
        return 0;
    }

    int Graph::update(GraphConfig update_config) {
        JsonParam option = update_config.get_option();
        std::vector<JsonParam> nodes_opt;
        option.get_object_list("nodes", nodes_opt);

        std::vector<NodeConfig> nodes_add;
        std::vector<NodeConfig> nodes_remove;
        std::vector<NodeConfig> nodes_reset;

        for (auto &node_config:update_config.get_nodes()) {
            std::string action = node_config.get_action();
            if (action == "add")
                nodes_add.push_back(node_config);
            else if (action == "remove")
                nodes_remove.push_back(node_config);
            else if (action == "reset")
                nodes_reset.push_back(node_config);
        }

        //dynamical add
        if (nodes_add.size()) {
            //find the node config of add
            std::map<int, std::shared_ptr<Node>> added_nodes;
            std::vector<std::shared_ptr<Node>> added_src_nodes;
            for (auto &node_config:nodes_add) {
                std::shared_ptr<Module> module_pre_allocated;
                int node_id = node_config.get_id();
                if (pre_modules_.count(node_id) > 0)
                    module_pre_allocated = pre_modules_[node_id];
                std::shared_ptr<Node> node;
                NodeCallBack callback;
                callback.get_node = std::bind(&Graph::get_node, this, std::placeholders::_1, std::placeholders::_2);
                callback.throttled_cb = std::bind(&Scheduler::add_or_remove_node, scheduler_, std::placeholders::_1,
                                                  std::placeholders::_2);
                callback.scheduler_cb = std::bind(&Scheduler::schedule_node, scheduler_, std::placeholders::_1);
                callback.clear_cb = std::bind(&Scheduler::clear_task, scheduler_, std::placeholders::_1,
                                        std::placeholders::_2);
                if (!callback_bindings_.count(node_id))
                    callback_bindings_[node_id] = std::make_shared<ModuleCallbackLayer>();
                node = std::make_shared<Node>(node_id, node_config, callback, module_pre_allocated, mode_,
                                              callback_bindings_[node_id]);

                if (node_config.get_scheduler() < scheduler_count_) {
                    node->set_scheduler_queue_id((node_config.get_scheduler()));
                    BMFLOG(BMF_INFO) << "node:" << node->get_type() << " " << node->get_id() << " scheduler "
                                     << node_config.get_scheduler();
                } else {
                    node->set_scheduler_queue_id(0);
                    BMFLOG(BMF_INFO) << "node:" << node->get_type() << " " << node->get_id() << " scheduler " << 0;
                }
                nodes_[node_config.get_id()] = node;
                added_nodes[node_config.get_id()] = node;

                if ((node_config.get_input_streams().size()) == 0) {
                    source_nodes_.push_back(node);
                    added_src_nodes.push_back(node);
                }
            }

            // create connections and pause relevant orginal nodes
            for (auto &node_iter:added_nodes) {
                std::map<int, std::shared_ptr<OutputStream>> output_streams;
                node_iter.second->get_output_streams(output_streams);
                // find all downstream connections for every output stream
                for (auto &output_stream:output_streams) {
                    bool b_matched;
                    b_matched = false;
                    for (auto node_iter:added_nodes) {
                        if (not node_iter.second->is_source()) {
                            std::shared_ptr<InputStreamManager> input_stream_manager;
                            node_iter.second->get_input_stream_manager(input_stream_manager);
                            for (auto input_stream:input_stream_manager->input_streams_) {
                                if (output_stream.second->identifier_ == input_stream.second->identifier_) {
                                    output_stream.second->add_mirror_stream(input_stream_manager,
                                                                            input_stream.first);
                                    input_stream.second->set_connected(true);
                                    b_matched = true;
                                    break;
                                }
                            }
                        }
                    }
                    if (not b_matched) {
                        //connect added inputs/outputs with original graph
                        //by check the not connected inputs/outputs
                        std::string prefix_name = output_stream.second->identifier_.substr(0,
                                                  output_stream.second->identifier_.find_first_of("."));
                        for (auto &node:nodes_) {
                            //find the matched link in original nodes
                            if (not node.second->is_source()) {
                                std::shared_ptr<InputStreamManager> input_stream_manager;
                                //if the new node output name prefix point to the node alias
                                if (prefix_name == node.second->get_alias()) {
                                    node.second->get_input_stream_manager(input_stream_manager);
                                    int new_id = input_stream_manager->add_stream(output_stream.second->identifier_,
                                                                                 node.second->get_id());
                                    output_stream.second->add_mirror_stream(input_stream_manager,
                                                                            new_id);
                                    input_stream_manager->input_streams_[new_id]->set_connected(true);
                                    BMFLOG(BMF_INFO) << "adding node " << node_iter.second->get_type()
                                                     << ", downstream: "<< prefix_name
                                                     << ", as input of " << output_stream.second->identifier_;
                                }
                            }
                        }
                    }
                }
            }

            //for upstream connections
            for (auto &node_iter:added_nodes) {
                std::shared_ptr<InputStreamManager> input_stream_manager;
                node_iter.second->get_input_stream_manager(input_stream_manager);
                //bool b_matched = false;
                for (auto input_stream:input_stream_manager->input_streams_) {
                    if (not input_stream.second->is_connected()) {
                        std::string prefix_name = input_stream.second->identifier_.substr(0,
                                                  input_stream.second->identifier_.find_first_of("."));
                        for (auto &node:nodes_) {
                            if (prefix_name == node.second->get_alias()) {
                                std::shared_ptr<OutputStreamManager> output_stream_manager;
                                node.second->get_output_stream_manager(output_stream_manager);
                                int new_id = output_stream_manager->add_stream(input_stream.second->identifier_);
                                output_stream_manager->output_streams_[new_id]->add_mirror_stream(
                                                                                input_stream_manager,
                                                                                input_stream.second->get_id());
                                input_stream_manager->input_streams_[input_stream.second->get_id()]->
                                                      set_connected(true);
                                node.second->set_outputstream_updated(true);
                                //b_matched = true;
                                BMFLOG(BMF_INFO) << "adding node " << node_iter.second->get_type()
                                                 << ", upstream: "<< prefix_name << ", as output of "
                                                 << input_stream.second->identifier_;
                            }
                        }
                    }
                    //if (not b_matched)
                    //    orphan_streams_.push_back(input_stream.second);
                }
            }

            for (auto &node:added_src_nodes)
                scheduler_->add_or_remove_node(node->get_id(), true);
        }

        //dynamical remove
        if (nodes_remove.size()) {
            for (auto &node_config:nodes_remove) {
                int id_of_rm_node = -1;
                for (auto &node:nodes_) {
                    if (node.second->get_alias() == node_config.get_alias()) {
                        BMFLOG(BMF_INFO) << "found the node to be removed: " << node.second->get_alias();
                        id_of_rm_node = node.first;
                        break;
                    }
                }
                if (id_of_rm_node == -1) {
                    BMFLOG(BMF_ERROR) << "cannot find the node to be removed";
                    return -1;
                }
                std::shared_ptr<Node> rm_node = nodes_[id_of_rm_node];

                std::vector<std::shared_ptr<Node>> paused_nodes;
                std::shared_ptr<InputStreamManager> input_stream_manager;
                std::shared_ptr<OutputStreamManager> output_stream_manager;
                std::vector<std::shared_ptr<OutputStream>> upstream_output_streams;
                rm_node->get_input_stream_manager(input_stream_manager);
                rm_node->get_output_stream_manager(output_stream_manager);
                if (not rm_node->is_source()) {
                    for (auto input_stream:input_stream_manager->input_streams_) {
                        bool b_matched = false;
                        if (input_stream.second->is_connected()) {
                            for (auto &node:nodes_) {
                                std::map<int, std::shared_ptr<OutputStream>> output_streams;
                                node.second->get_output_streams(output_streams);
                                for (auto output_stream:output_streams) {
                                    if (output_stream.second->identifier_ == input_stream.second->identifier_) {
                                        //got the connected upstream
                                        BMFLOG(BMF_INFO) << "found the upstream node: "
                                                         << node.second->get_type() << std::endl;
                                        bool already_paused = false;
                                        for (auto pn:paused_nodes) {
                                            if (pn->get_id() == node.second->get_id()) {
                                                already_paused = true;
                                                break;
                                            }
                                        }
                                        if (already_paused == false) {
                                            node.second->wait_paused();
                                            paused_nodes.push_back(node.second);
                                        }
                                        upstream_output_streams.push_back(output_streams[output_stream.first]);
                                        break;
                                    }
                                }
                            }
                        }
                    }

                    //probe on the output about the specail EOF signal, avoid to passdown
                    output_stream_manager->probe_eof();

                    //push EOF into the input streams of the node
                    for (auto stream:input_stream_manager->input_streams_) {
                        std::shared_ptr<SafeQueue<Packet> > q = std::make_shared<SafeQueue<Packet>>();
                        BMFLOG(BMF_INFO) << "push eof to inputstream of removed node: " << stream.second->identifier_;
                        q->push(Packet::generate_eof_packet());
                        stream.second->add_packets(q);
                    }

                    //wait utile the EOF was consumed
                    for (auto input_stream:input_stream_manager->input_streams_)
                        input_stream_manager->wait_on_stream_empty(input_stream.first);

                    rm_node->wait_paused();

                    //remove the upstream after output stream consumed
                    for (auto node:paused_nodes) {
                        std::shared_ptr<OutputStreamManager> out_str_mng;
                        std::map<int, std::shared_ptr<OutputStream>> output_streams;
                        node->get_output_stream_manager(out_str_mng);
                        node->get_output_streams(output_streams);
                        for (auto &output_stream:output_streams) {
                            for (auto rm_stream:upstream_output_streams) {
                                if (output_stream.second->identifier_ == rm_stream->identifier_) {
                                    out_str_mng->remove_stream(output_stream.first, rm_stream->stream_id_);
                                    BMFLOG(BMF_INFO) << "remove stream: " << rm_stream->identifier_
                                                     << "stream id: " << rm_stream->stream_id_;
                                    //make sure to notify the output streams of the upstream nodes are changed
                                    node->set_outputstream_updated(true);
                                    rm_stream->stream_id_ = -1;//removed tag
                                    break;
                                }
                            }
                        }
                    }
                } else {
                    rm_node->wait_paused();

                    output_stream_manager->probe_eof();

                    for (auto output_stream:output_stream_manager->output_streams_) {
                        std::shared_ptr<SafeQueue<Packet> > q = std::make_shared<SafeQueue<Packet>>();
                        BMFLOG(BMF_INFO) << "push eof to outputstream of removing node";
                        q->push(Packet::generate_eof_packet());
                        output_stream.second->propagate_packets(q);
                    }
                }

                for (auto node:paused_nodes) {
                    BMFLOG(BMF_INFO) << "paused us node recover: " << node->get_id() << " alias: "
                                     << node->get_alias();
                    node->set_status(NodeState::PENDING);
                }

                //remove the downstream
                for (auto output_stream:output_stream_manager->output_streams_)
                    output_stream_manager->wait_on_stream_empty(output_stream.first);
                //pause the down stream nodes
                std::vector<int> ds_nodes_id;
                output_stream_manager->get_outlink_nodes_id(ds_nodes_id);
                int ret;
                std::vector<std::shared_ptr<Node>> paused_ds_nodes;
                for (int i = 0 ; i < ds_nodes_id.size(); i++) {
                    std::shared_ptr<Node> nd;
                    if (get_node(ds_nodes_id[i], nd) == 0) {
                        nd->wait_paused();
                        paused_ds_nodes.push_back(nd);
                    } else {
                        BMFLOG(BMF_ERROR) << "down stream node can't be got: " << i;
                        return -1;
                    }
                }

                std::vector<int> rm_streams_id;
                for (auto output_stream:output_stream_manager->output_streams_)
                    rm_streams_id.push_back(output_stream.first);
                for (auto streams_id:rm_streams_id) {
                    BMFLOG(BMF_INFO) << "remove down stream: "
                                     << output_stream_manager->output_streams_[streams_id]->identifier_;
                    output_stream_manager->remove_stream(streams_id, -1);
                }

                //remove the nodes in scheduler
                while (scheduler_->nodes_to_schedule_.find(rm_node->get_id()) !=
                       scheduler_->nodes_to_schedule_.end())
                    scheduler_->add_or_remove_node(rm_node->get_id(), false);

                rm_node->close();
                nodes_.erase(rm_node->get_id());
                BMFLOG(BMF_INFO) << "remove node: " << rm_node->get_id() << " alias: " << rm_node->get_alias();

                for (auto node:paused_ds_nodes) {
                    BMFLOG(BMF_INFO) << "paused ds node recover: " << node->get_id() << " alias: "
                                     << node->get_alias();
                    std::shared_ptr<InputStreamManager> input_stream_manager;
                    node->get_input_stream_manager(input_stream_manager);
                    if (input_stream_manager->input_streams_.size() == 0) {
                        node->set_source(true);
                        BMFLOG(BMF_INFO) << "set source: id " << node->get_id() << " " << node->get_alias();
                        scheduler_->add_or_remove_node(node->get_id(), true);
                    } else
                        node->set_status(NodeState::PENDING);
                }
            }
        }

        if (nodes_reset.size()) {
            for (auto &node_config:nodes_reset) {
                int id_of_reset_node = -1;
                for (auto &node:nodes_) {
                    if (node.second->get_alias() == node_config.get_alias()) {
                        BMFLOG(BMF_INFO) << "found the node to be reset: " << node.second->get_alias();
                        id_of_reset_node = node.first;
                        break;
                    }
                }
                if (id_of_reset_node == -1) {
                    BMFLOG(BMF_ERROR) << "cannot find the node to be removed";
                    return -1;
                }
                std::shared_ptr<Node> reset_node = nodes_[id_of_reset_node];

                //update the option
                reset_node->need_opt_reset(node_config.get_option());
            }
        }

        return 0;
    }

    bool Graph::all_nodes_done() {
        for (auto &node_iter: nodes_) {
            if (not node_iter.second->is_closed()) {
                return false;
            }
        }
        return true;
    }

    int Graph::close() {
        std::unique_lock<std::mutex> lk(con_var_mutex_);
        //if (not all_nodes_done())
        if (closed_count_ != nodes_.size())
            cond_close_.wait(lk);

        scheduler_->close();
        g_ptr.clear();
        if (scheduler_->eptr_){
            std::rethrow_exception(scheduler_->eptr_);
        }
        return 0;
    }

    int Graph::get_node(int node_id, std::shared_ptr<Node> &node) {
        if (nodes_.count(node_id)) {
            node = nodes_[node_id];
            return 0;
        }
        return -1;
    }

    int Graph::force_close() {
        for (auto &node :nodes_) {
            node.second->close();
        }
        scheduler_->close();
        return 0;
    }

    // manually insert C++ packet to graph
    int Graph::add_input_stream_packet(std::string const &stream_name, Packet &packet, bool block) {
        if (input_streams_.count(stream_name) > 0) {
            if (block){
                
                while (input_streams_[stream_name]->manager_->any_of_downstream_full()){
                    {
                        usleep(1000);
                    }
                }
            }
            input_streams_[stream_name]->add_packet(packet);
        }
        return 0;
    }

    // manually poll output packet and return C++ packet
    Packet Graph::poll_output_stream_packet(std::string const &stream_name, bool block) {
        Packet packet;
        if (output_streams_.count(stream_name) > 0) {
            output_streams_[stream_name]->poll_packet(packet, block);
        }
        return packet;
    }

//TODO manually insert a eos packet to indicate the graph input stream is done
    int Graph::add_eos_packet(std::string const &stream_name) {
        if (input_streams_.count(stream_name) > 0) {
            Packet packet = Packet::generate_eos_packet();
            input_streams_[stream_name]->add_packet(packet);
        }
        return 0;
    }

    void Graph::pause_running(double_t timeout) {
        if (paused_)
            return;
        scheduler_->pause();
        paused_ = true;
        if (timeout > 0) {
            auto f = [](Graph *g, int timout) {
                usleep(timout);
                g->resume_running();
            };
            std::thread(f, this, timeout * 1000);
        }
    }

    void Graph::resume_running() {
        if (!paused_)
            return;
        scheduler_->resume();
        paused_ = false;
    }

    void Graph::print_node_info_pretty() {
#define LEFTW(width) std::setiosflags(std::ios::left)<<std::setw(width)

        std::cerr << LEFTW(10) << "NODE" << LEFTW(30) << "TYPE" << LEFTW(10) << "STATUS"
                  << LEFTW(20) << "SCHEDULE_COUNT" << LEFTW(20) << "SCHEDULE_SUCCESS" << LEFTW(20) << "TIMESTAMP"
                  << LEFTW(10) << "IS_SOURCE" << std::endl;
        for (auto nd:nodes_) {
            std::cerr << LEFTW(10) << nd.second->get_id() << LEFTW(30) << nd.second->get_type() << LEFTW(10)
                      << nd.second->get_status()
                      << LEFTW(20) << nd.second->get_schedule_attempt_cnt() << LEFTW(20)
                      << nd.second->get_schedule_success_cnt() << LEFTW(20) << nd.second->get_last_timestamp()
                      << LEFTW(10) << (nd.second->is_source() ? "YES" : "NO") << std::endl;
        }
#undef LEFTW
    }

    void Graph::quit_gracefully() {
        std::cerr << "quitting..." << std::endl;
        for (auto g:g_ptr) {
            g->print_node_info_pretty();
            g->force_close();
        }
    }

    Graph::~Graph() {
        scheduler_->close();
    }

    bmf::GraphRunningInfo Graph::status() {
        return RunningInfoCollector().collect_graph_info(this);
    }

    void GraphInputStream::set_manager(std::shared_ptr<OutputStreamManager> &manager) {
        manager_ = manager;
    }

    void GraphInputStream::add_packet(Packet &packet) {
        std::shared_ptr<SafeQueue<Packet> > packets = std::make_shared<SafeQueue<Packet> >();
        packets->push(packet);
        manager_->propagate_packets(0, packets);
    }

    void GraphOutputStream::set_manager(std::shared_ptr<InputStreamManager> &input_manager) {
        input_manager_ = input_manager;
    }

    void GraphOutputStream::poll_packet(Packet &packet, bool block) {
        packet = input_manager_->pop_next_packet(0, block);
    }

END_BMF_ENGINE_NS
