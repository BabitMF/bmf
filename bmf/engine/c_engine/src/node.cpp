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

#include "../include/node.h"
#include "../include/input_stream_manager.h"
#include "../include/module_factory.h"
#include "../include/callback_layer.h"

#include <bmf/sdk/log.h>

#include <functional>
#include <memory>

#include <bmf/sdk/trace.h>

BEGIN_BMF_ENGINE_NS
    USE_BMF_SDK_NS

    Node::Node(int node_id, NodeConfig &node_config, NodeCallBack &node_callback,
               std::shared_ptr<Module> pre_allocated_module, BmfMode mode,
               std::shared_ptr<ModuleCallbackLayer> callbacks)
            : id_(node_id), node_config_(node_config), callback_(node_callback), mode_(mode),
              module_callbacks_(callbacks) {
        type_ = node_config_.get_module_info().module_name;
        module_name_ = type_;
        node_name_ = "Node_" + std::to_string(id_) + "_" + module_name_;

#ifndef NO_TRACE
        TraceProcessEmitter trace_emitter = TraceProcessEmitter(PROCESSING, node_name_);
#endif

        is_source_ = node_config.input_streams.empty();
        queue_size_limit_ = node_config_.get_node_meta().get_queue_size_limit();
        // pending task means the task has been added to scheduler queue
        // but haven't been executed, for source node, we need use this
        // value to control task filling speed
        pending_tasks_ = 0;
        //max_pending_tasks_ = 5;
        max_pending_tasks_ = queue_size_limit_;
        printf("debug queue size, node %d, queue size: %d\n", node_id, queue_size_limit_);
        task_processed_cnt_ = 0;
        is_premodule_ = false;

        if (pre_allocated_module == nullptr) {
            is_premodule_ = false;
            JsonParam node_option_param = node_config_.get_option();
            module_info_ = ModuleFactory::create_module(type_, id_, node_option_param, node_config_.module.module_type,
                                                        node_config_.module.module_path,
                                                        node_config_.module.module_entry, module_);

            BMF_TRACE_PROCESS(module_name_.c_str(), "init", START);
            module_->init();
            BMF_TRACE_PROCESS(module_name_.c_str(), "init", END);
        } else {
            module_ = pre_allocated_module;
            //for real premodule coming from user
            if (node_config.get_node_meta().get_premodule_id() > 0) {
                is_premodule_ = true;
                module_->reset();
            }
            else
                module_->init();
            module_->node_id_ = node_id;
        }

        module_->set_callback([this](int64_t key, CBytes para) -> CBytes {
            return this->module_callbacks_->call(key, para);
        });

        // some special nodes can keep producing outputs even no inputs
        // e.g. loop, for this kind of node, it will be closed only if all
        // downstream nodes are closed, and the task priority of this node
        // should be set to low
        infinity_node_ = module_->is_infinity();

        output_stream_manager_ = std::make_shared<OutputStreamManager>(node_config.get_output_streams());

        InputStreamManagerCallBack callback;
        callback.scheduler_cb = callback_.scheduler_cb;
        callback.throttled_cb = callback_.throttled_cb;
        callback.sched_required = callback_.sched_required;
        callback.get_node_cb = callback_.get_node;
        callback.notify_cb = [this]() -> bool {
            return this->schedule_node();
        };
        callback.node_is_closed_cb = [this]() -> bool {
            return this->is_closed();
        };

        create_input_stream_manager(node_config.get_input_manager(), node_id, node_config.get_input_streams(),
                                    output_stream_manager_->get_stream_id_list(), callback, queue_size_limit_,
                                    input_stream_manager_);

        //register hungry_check
        for (auto stream_id : input_stream_manager_->stream_id_list_) {
            if (module_->need_hungry_check(stream_id)) {
                hungry_check_func_[stream_id].push_back([this, stream_id]() -> bool {
                    return this->module_->is_hungry(stream_id);
                });
            }
        }

        force_close_ = false;
        if (!node_config.get_output_streams().empty()) {
            force_close_ = true;
        }

        state_ = NodeState::RUNNING;
        schedule_node_cnt_ = 0;
        schedule_node_success_cnt_ = 0;
        source_timestamp_ = 0;
        last_timestamp_ = 0;

        wait_pause_ = false;
        need_opt_reset_ = false;
    }

    int Node::inc_pending_task() {
        mutex_.lock();
        pending_tasks_++;
        mutex_.unlock();
        return 0;
    }

    int Node::dec_pending_task() {
        mutex_.lock();
        pending_tasks_--;
        mutex_.unlock();
        return 0;
    }

    bool Node::is_source() {
        return is_source_;
    }

    int Node::get_id() {
        return id_;
    }

    std::string Node::get_alias() {
        return node_config_.get_alias();
    }

    std::string Node::get_action() {
        return node_config_.get_action();
    }

    void Node::set_source(bool flag) {
        is_source_ = flag;
    }

    int Node::close() {
        mutex_.lock();
        //callback_.throttled_cb(id_, false);
        for (auto &input_stream:input_stream_manager_->input_streams_)
            if (input_stream.second->is_full())
                input_stream.second->clear_queue();
        if (!is_premodule_){
            module_->close();
        }
        state_ = NodeState::CLOSED;
        BMFLOG_NODE(BMF_INFO, id_) << "close node";
        callback_.sched_required(id_, true);
        mutex_.unlock();
        return 0;
    }

    int Node::reset() {
        mutex_.lock();
        // reset module of this node
        module_->reset();
        // remove from source nodes
        set_source(false);

        // clear remaining node task from schedule queue
        callback_.clear_cb(id_, get_scheduler_queue_id());

        // reset all input_streams of node
        for (auto &input_stream : input_stream_manager_->input_streams_) {
            if (input_stream.second->get_block()) {
                input_stream.second->set_block(false);
            }
        }

        if (!all_input_queue_empty()) {//in some case such as server mode previous blocked
            int res = input_stream_manager_->schedule_node();
            if (res)
                schedule_node_success_cnt_++;
        }

        mutex_.unlock();
        return 0;
    }

    bool Node::is_closed() {
        return state_ == NodeState::CLOSED;
    }

    void Node::register_hungry_check_func(int input_idx, std::function<bool()> &func) {
        hungry_check_func_[input_idx].push_back(func);
    }

    void Node::get_hungry_check_func(int input_idx, std::vector<std::function<bool()> > &funcs) {
        if (hungry_check_func_.count(input_idx)) {
            funcs = hungry_check_func_[input_idx];
        }
    }

    bool Node::is_hungry() {
        if (hungry_check_func_.empty()) {
            return true;
        }
        for (auto &hungry_check_func:hungry_check_func_) {
            for (auto &func:hungry_check_func.second) {
                if (func()) {
                    return true;
                }
            }
        }
        return false;
    }

    int64_t Node::get_source_timestamp() {
        source_timestamp_++;
        return source_timestamp_;
    }

    bool Node::any_of_downstream_full() {

        return output_stream_manager_->any_of_downstream_full();

    }

    bool Node::any_of_input_queue_full() {
        for (auto input_stream = input_stream_manager_->input_streams_.begin();
             input_stream != input_stream_manager_->input_streams_.end(); input_stream++) {
            if (input_stream->second->is_full())
                return true;
        }
        return false;
    }

    bool Node::all_input_queue_empty() {
        for (auto input_stream = input_stream_manager_->input_streams_.begin();
             input_stream != input_stream_manager_->input_streams_.end(); input_stream++) {
            if (!input_stream->second->is_empty())
                return false;
        }
        return true;
    }

    bool Node::too_many_tasks_pending() {
        return pending_tasks_ >= max_pending_tasks_;
    }

    int Node::all_downstream_nodes_closed() {
        for (auto &output_stream:output_stream_manager_->output_streams_) {
            for (auto &mirror_stream : output_stream.second->mirror_streams_) {
                int node_id = mirror_stream.input_stream_manager_->node_id_;
                std::shared_ptr<Node> node = nullptr;
                callback_.get_node(node_id, node);
                if (node == nullptr or not node->is_closed()) {
                    return false;
                }
            }
        }
        return true;
    }

    int Node::need_force_close() {
        return force_close_;
    }

    int Node::need_opt_reset(JsonParam reset_opt) {
        opt_reset_mutex_.lock();
        need_opt_reset_ = true;
        reset_option_ = reset_opt;
        BMFLOG(BMF_INFO) << "need_opt_reset option: " << reset_opt.dump();
        opt_reset_mutex_.unlock();
        return 0;
    }

// In current design, this function will not be called in parallel
// since one node is only scheduled by one scheduler queue
    int Node::process_node(Task &task) {
        if (state_ == NodeState::CLOSED || state_ == NodeState::PAUSE_DONE) {
            dec_pending_task();
            return 0;
        }

#ifndef NO_TRACE
        TraceProcessEmitter trace_emitter = TraceProcessEmitter(PROCESSING, node_name_);
#endif

        //TODO check the task is valid

        last_timestamp_ = task.timestamp();
//        std::cout<<"timestamp="<<last_timestamp_<<std::endl;
        //task timestamp is eof means all inputs are done, but at this moment
        //the outputs might not be done, so we can schedule the node as source
        //(no inputs, like source node) until all outputs done
        if (not is_source() and task.timestamp() == BMF_EOF) {
            BMFLOG_NODE(BMF_INFO, id_) << "process eof, add node to scheduler";
            // become source node
            set_source(true);
            // add to scheduler
            //callback_.throttled_cb(id_, true);
            //callback_.sched_required(id_, false);
        }
            // used only for SERVER mode
            // receive eos means that the node is ready to close
            // before close we should propagate eos pkt to the downstream
        else if (task.timestamp() == EOS) {
            dec_pending_task();
            BMFLOG_NODE(BMF_INFO, id_) << "meet eos, close the node and propagate eos pkt instead of node process";
            for (auto &iter : task.get_outputs()) {
                iter.second->push(Packet::generate_eos_packet());
            }
            output_stream_manager_->post_process(task);
            close();
            return 0;
        }

        auto cleanup = [this] {
            this->task_processed_cnt_++;
            this->dec_pending_task();
            BMFLOG_NODE(BMF_ERROR, this->id_) << "Process node failed, will exit.";
        };

        int result = 0;
        try {
            opt_reset_mutex_.lock();
            if (need_opt_reset_) {
                module_->dynamic_reset(reset_option_);
                need_opt_reset_ = false;
            }
            opt_reset_mutex_.unlock();

            BMF_TRACE_PROCESS(module_name_.c_str(), "process", START);
            result = module_->process(task);
            BMF_TRACE_PROCESS(module_name_.c_str(), "process", END);
            if (result != 0)
                BMF_Error_(BMF_StsBadArg, "[%s] Process result != 0.\n", node_name_.c_str());
        }
        //catch (boost::python::error_already_set const &) {
        //    std::cerr << "python module exec failed" << std::endl;
        //    PyErr_Print();
        //    cleanup();
        //    throw std::runtime_error("[" + node_name_ + "] " + "Python error already set.");
        //}
        catch (...) {
            cleanup();
            std::rethrow_exception(std::current_exception());
        }

        task_processed_cnt_++;
        if (task.timestamp() == DONE && not is_closed()) {
            BMFLOG_NODE(BMF_INFO, id_) << "Process node end";
            if (mode_ == BmfMode::SERVER_MODE) {
                reset();
            } else {
                close();
            }
        }

        // call post process, propagate packets to downstream node
        output_stream_manager_->post_process(task);

        dec_pending_task();

        if ((wait_pause_ == true && pending_tasks_ == 0)) {
            state_ = NodeState::PAUSE_DONE;
            pause_event_.notify_all();
            return 0;
        }

        bool is_blocked = false;
        if (mode_ == BmfMode::SERVER_MODE && !is_source()) {
            uint8_t blk_num = 0;
            for (auto &input_stream : input_stream_manager_->input_streams_) {
                if (input_stream.second->get_block()) {
                    blk_num++;
                }
            }
            if (blk_num == input_stream_manager_->input_streams_.size())
                is_blocked = true;
        }

        //BMFLOG_NODE(BMF_INFO, id_) << "block: " << is_blocked << "state: " << get_status();

        if (!is_blocked)
            if (state_ != NodeState::CLOSED)
                callback_.sched_required(id_, false);

        return 0;
    }

/*
There are two cases which will run into schedule_node, so we need a mutex here
1. while an input packet is added to input_stream, this is called by input_stream_manager
2. while node is in PENDING or all input streams done, it's scheduled by scheduler
*/
    bool Node::schedule_node() {
#ifndef NO_TRACE
        TraceProcessEmitter trace_emitter = TraceProcessEmitter(SCHEDULE, node_name_);
#endif
        mutex_.lock();
//        BMFLOG_NODE(BMF_INFO, id_) << "scheduling...";
        schedule_node_cnt_++;
        if (state_ == NodeState::PAUSE_DONE) {
            mutex_.unlock();
            return false;
        }
        //we don't need schedule node if it's closed or set to pause
        if (state_ == NodeState::CLOSED || (wait_pause_ == true && pending_tasks_ != 0)) {
            mutex_.unlock();
            return false;
        }
        // for node that need force_close, check if all downstream nodes are closed already
        // if yes, set it to closed, and remove it from node scheduler
        if (need_force_close()) {
            if (all_downstream_nodes_closed()) {
                close();
                mutex_.unlock();
                BMFLOG_NODE(BMF_INFO, id_) << "scheduling failed, all downstream node closed: " << type_;
                return false;
            }
        }

        bool result = false;
        if (is_source()) {
            Task task = Task(id_, input_stream_manager_->stream_id_list_, output_stream_manager_->get_stream_id_list());
            if (infinity_node_) {
                task.set_timestamp(INF_SRC);
            } else {
                task.set_timestamp(get_source_timestamp());
            }
            callback_.scheduler_cb(task);
            schedule_node_success_cnt_++;
            result = true;
        } else {
            if (node_output_updated_) {
                input_stream_manager_->output_stream_id_list_ = output_stream_manager_->get_stream_id_list();
                node_output_updated_ = false;
            }
            result = input_stream_manager_->schedule_node();
            if (result)
                schedule_node_success_cnt_++;
        }
        mutex_.unlock();
        return result;
    }

    void Node::wait_paused() {
        std::unique_lock<std::mutex> lk(pause_mutex_);
        wait_pause_ = true;
        while (state_ != NodeState::PAUSE_DONE && state_ != NodeState::CLOSED) {
            pause_event_.wait_for(lk, std::chrono::microseconds(40));
            if (pending_tasks_ == 0) {
                BMFLOG_NODE(BMF_INFO, id_) << "wait pause: pending task is zero";
                state_ = NodeState::PAUSE_DONE;
                break;
            }
        }
        wait_pause_ = false;
    }

    void Node::set_scheduler_queue_id(int scheduler_queue_id) {
        scheduler_queue_id_ = scheduler_queue_id;
    }

    int Node::get_scheduler_queue_id() {
        return scheduler_queue_id_;
    }

    int Node::get_input_stream_manager(std::shared_ptr<InputStreamManager> &input_stream_manager) {
        input_stream_manager = input_stream_manager_;
        return 0;
    }

    int Node::get_input_streams(std::map<int, std::shared_ptr<InputStream> > &input_streams) {
        if (input_stream_manager_ != NULL) {
            input_streams = input_stream_manager_->input_streams_;
        }
        return 0;
    }

    int Node::get_output_stream_manager(std::shared_ptr<OutputStreamManager> &output_stream_manager) {
        output_stream_manager = output_stream_manager_;
        return 0;
    }

    int Node::get_output_streams(std::map<int, std::shared_ptr<OutputStream>> &output_streams) {
        output_streams = output_stream_manager_->output_streams_;
        return 0;
    }

    std::string Node::get_type() {
        return type_;
    }

    std::string Node::get_status() {
        switch (state_) {
            case NodeState::PENDING:
                return "PENDING";
            case NodeState::NOT_INITED:
                return "NOT_INITED";
            case NodeState::RUNNING:
                return "RUNNING";
            case NodeState::CLOSED:
                return "CLOSE";
            case NodeState::PAUSE_DONE:
                return "PAUSE_DONE";
        }
        return "UNKNOWN";
    }

    void Node::set_status(NodeState state) {
        mutex_.lock();
        state_ = state;
        mutex_.unlock();
    }

    void Node::set_outputstream_updated(bool update) {
        node_output_updated_ = update;
    }

    int Node::get_schedule_attempt_cnt() {
        return schedule_node_cnt_;
    }

    int Node::get_schedule_success_cnt() {
        return schedule_node_success_cnt_;
    }

    int64_t Node::get_last_timestamp() {
        return last_timestamp_;
    }
END_BMF_ENGINE_NS
