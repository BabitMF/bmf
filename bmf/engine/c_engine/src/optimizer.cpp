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

#include "../include/optimizer.h"


BEGIN_BMF_ENGINE_NS
    USE_BMF_SDK_NS

    namespace Optimizer {
        void convert_filter_para(NodeConfig &node) {
            json new_option;
            json f;
            f["inputs"] = json(std::vector<std::string>());
            f["outputs"] = json(std::vector<std::string>());

            for (int index = 0; index < node.get_input_streams().size(); index++) {
                json input_pin;
                input_pin["stream"] = node.get_input_streams()[index].get_identifier();
                input_pin["pin"] = index;
                f["inputs"].push_back(input_pin);
            }

            for (int index = 0; index < node.get_output_streams().size(); index++) {
                json output_pin;
                output_pin["stream"] = node.get_output_streams()[index].get_identifier();
                output_pin["pin"] = index;
                f["outputs"].push_back(output_pin);
            }

            std::string name;
            node.get_option().get_string("name", name);
            f["name"] = name;

            if (node.get_option().has_key("para")) {
                std::string para;
                node.get_option().get_string("para", para);
                f["para"] = para;
            }

            new_option["filters"].push_back(f);
            node.set_option(JsonParam(new_option));
        }

        void convert_filter_para_for_graph(std::vector<NodeConfig> &nodes) {
            for (NodeConfig &node:nodes) {
                if (node.get_module_info().get_module_name() == "c_ffmpeg_filter") {
                    convert_filter_para(node);
                }
            }
        }

        int find_merged_link(json &links, StreamConfig stream) {
            int pin = -1;
            json link_remove;

            for (json link:links) {
                if (link["stream"] == stream.get_identifier()) {
                    pin = link["pin"];
                    link_remove = link;
                    break;
                }
            }

            if (link_remove.size() > 0) {
                links.erase(std::remove(links.begin(), links.end(), link_remove), links.end());
            }

            return pin;
        }

        void replace_stream_name_with_id(NodeConfig &node) {
            json option = node.get_option().json_value_;
            json &filter_option = option["filters"];
            bool stream_found;

            for (int index = 0; index < node.get_input_streams().size(); index++) {
                stream_found = false;
                for (json &f:filter_option) {
                    if (f.find("inputs") != f.end()) {
                        for (json &input_pin:f["inputs"]) {
                            if (input_pin["stream"] == node.get_input_streams()[index].get_identifier()) {
                                input_pin["stream"] = index;
                                stream_found = true;
                                break;
                            }
                        }
                    }
                    if (stream_found)
                        break;
                }
            }

            for (int index = 0; index < node.get_output_streams().size(); index++) {
                stream_found = false;
                for (json &f:filter_option) {
                    if (f.find("outputs") != f.end()) {
                        for (json &output_pin:f["outputs"]) {
                            if (output_pin["stream"] == node.get_output_streams()[index].get_identifier()) {
                                output_pin["stream"] = index;
                                stream_found = true;
                                break;
                            }
                        }
                    }
                    if (stream_found)
                        break;
                }
            }

            node.set_option(JsonParam(option));
        }

        void replace_stream_name_for_graph(std::vector<NodeConfig> &nodes) {
            for (NodeConfig &node:nodes) {
                if (node.get_module_info().get_module_name() == "c_ffmpeg_filter") {
                    replace_stream_name_with_id(node);
                }
            }
        }

        void merge_two_nodes(NodeConfig &n1, NodeConfig &n2) {
            for (StreamConfig input_stream:n2.get_input_streams()) {
                n1.add_input_stream(input_stream);
            }

            for (StreamConfig output_stream:n2.get_output_streams()) {
                n1.add_output_stream(output_stream);
            }

            json option = n1.get_option().json_value_;
            json filter_option_1 = option["filters"];
            json filter_option_2 = n2.get_option().json_value_["filters"];

            for (json filter:filter_option_2) {
                filter_option_1.push_back(filter);
            }

            std::vector<StreamConfig> removed_stream;
            std::vector<StreamConfig> &input_streams = n1.get_input_streams();
            std::vector<StreamConfig> &output_streams = n1.get_output_streams();

            for (StreamConfig input_stream:input_streams) {
                if (std::find(output_streams.begin(), output_streams.end(), input_stream) != output_streams.end()) {
                    removed_stream.push_back(input_stream);

                    int filter_id;
                    int out_pin;

                    for (int index = 0; index < filter_option_1.size(); index++) {
                        json &f = filter_option_1[index];
                        if (f.find("inputs") != f.end()) {
                            out_pin = find_merged_link(f["inputs"], input_stream);
                            if (out_pin != -1) {
                                filter_id = index;
                                break;
                            }
                        }
                    }

                    for (int index = 0; index < filter_option_1.size(); index++) {
                        json &f = filter_option_1[index];
                        if (f.find("outputs") != f.end()) {
                            int in_pin = find_merged_link(f["outputs"], input_stream);
                            if (in_pin != -1) {
                                json link;
                                link["input_pin"] = in_pin;
                                link["output_pin"] = out_pin;
                                link["output_filter"] = filter_id;
                                if (f.find("links") == f.end()) {
                                    f["links"] = json(std::vector<json>());
                                }
                                f["links"].push_back(link);
                            }
                        }
                    }
                }
            }

            for (StreamConfig stream:removed_stream) {
                if (std::find(input_streams.begin(), input_streams.end(), stream) != input_streams.end()) {
                    input_streams.erase(std::remove(input_streams.begin(), input_streams.end(), stream),
                                        input_streams.end());
                }
                if (std::find(output_streams.begin(), output_streams.end(), stream) != output_streams.end()) {
                    output_streams.erase(std::remove(output_streams.begin(), output_streams.end(), stream),
                                         output_streams.end());
                }
            }

            option["filters"] = filter_option_1;
            n1.set_option(JsonParam(option));
        }

        NodeConfig merge_ffmpeg_filter_nodes(std::vector<NodeConfig> &merge_nodes) {
            NodeConfig merge_node;
            if (merge_nodes.size() == 0) {
                return merge_node;
            }

            merge_node = merge_nodes[0];
            for (int i = 1; i < merge_nodes.size(); i++) {
                merge_two_nodes(merge_node, merge_nodes[i]);
            }

            return merge_node;
        }

        std::vector<Neighbour> find_all_neighbours(std::vector<NodeConfig> opt_nodes, NodeConfig merged_node) {
            std::vector<Neighbour> neighbours;

            for (StreamConfig output_stream:merged_node.get_output_streams()) {
                for (NodeConfig node:opt_nodes) {
                    std::vector<StreamConfig> input_streams = node.get_input_streams();
                    if (std::find(input_streams.begin(), input_streams.end(), output_stream) != input_streams.end()) {
                        Neighbour nb;
                        nb.node = node;
                        nb.root_stream = output_stream;
                        neighbours.push_back(nb);
                    }
                }
            }
            return neighbours;
        }

        StreamConfig
        has_circle(std::vector<NodeConfig> opt_nodes, NodeConfig merged_node, std::map<int, bool> &rec_stack) {
            StreamConfig s;
            rec_stack[merged_node.get_id()] = true;
            std::vector<Neighbour> neighbours = find_all_neighbours(opt_nodes, merged_node);

            for (Neighbour nb:neighbours) {
                if (rec_stack.count(nb.node.get_id()) == 0 || !rec_stack[nb.node.get_id()]) {
                    StreamConfig circle_stream = has_circle(opt_nodes, nb.node, rec_stack);
                    if (circle_stream.get_identifier().length() > 0) {
                        return circle_stream;
                    }
                } else {
                    return nb.root_stream;
                }
            }
            rec_stack[merged_node.get_id()] = false;
            return s;
        }

        StreamConfig find_first_circle_node(std::vector<NodeConfig> opt_nodes, NodeConfig merged_node) {
            std::map<int, bool> rec_stack;

            StreamConfig stream = has_circle(opt_nodes, merged_node, rec_stack);
            return stream;
        }

        void optimize(std::vector<NodeConfig> &nodes) {
            // nodes_done is used to record ffmpeg_filter node that already optimized
            std::vector<NodeConfig> nodes_done;

            while (true) {
                std::vector<NodeConfig> nodes_to_merge;

                // put all ffmpeg_filter nodes into nodes_to_merge and try to combine it to one node
                for (NodeConfig &node:nodes) {
                    if (node.get_module_info().get_module_name() == "c_ffmpeg_filter" &&
                        std::find(nodes_done.begin(), nodes_done.end(), node) == nodes_done.end()) {
                        nodes_to_merge.push_back(node);
                    }
                }

                for (NodeConfig node:nodes_to_merge) {
                    nodes.erase(std::remove(nodes.begin(), nodes.end(), node), nodes.end());
                }

                if (nodes_to_merge.size() == 0) {
                    break;
                }

                // nodes to merge should have the same scheduler id
                int scheduler_id = nodes_to_merge[0].get_scheduler();
                for (int i = 1; i < nodes_to_merge.size(); i++) {
                    NodeConfig node = nodes_to_merge[i];
                    if (node.get_scheduler() != scheduler_id){
                        // remove node from nodes_to_merge and add back to node list
                        nodes_to_merge.erase(std::remove(nodes_to_merge.begin(), nodes_to_merge.end(), node), nodes_to_merge.end());
                        nodes.push_back(node);
                        i--;
                    }
                }

                while (true) {
                    // do node merge
                    NodeConfig merged_node = merge_ffmpeg_filter_nodes(nodes_to_merge);
                    nodes.push_back(merged_node);

                    // check if has a circle
                    StreamConfig circle_stream = find_first_circle_node(nodes, merged_node);

                    if (circle_stream.get_identifier().length() > 0) {
                        NodeConfig circle_node;
                        // find circle-end node according to stream
                        for (NodeConfig node:nodes_to_merge) {
                            std::vector<StreamConfig> input_streams = node.get_input_streams();
                            if (std::find(input_streams.begin(), input_streams.end(), circle_stream) !=
                                input_streams.end()) {
                                circle_node = node;
                                break;
                            }
                        }

                        // remove it from nodes_to_merge and add it back to node list
                        nodes_to_merge.erase(std::remove(nodes_to_merge.begin(), nodes_to_merge.end(), circle_node),
                                             nodes_to_merge.end());
                        nodes.push_back(circle_node);
                        nodes.erase(std::remove(nodes.begin(), nodes.end(), merged_node), nodes.end());
                        continue;
                    } else {
                        nodes_done.push_back(merged_node);
                        break;
                    }
                }
            }
        }

        void merge_subgraph(GraphConfig &main_config, GraphConfig &sub_config, int sub_node_id) {
            NodeConfig sub_graph_node;

            // find subgraph node from main graph
            for (NodeConfig &node:main_config.nodes) {
                if (node.get_id() == sub_node_id) {
                    sub_graph_node = node;
                    break;
                }
            }

            // find source nodes from subgraph-inside-nodes and replace their input_streams
            for (int i = 0; i < sub_config.get_input_streams().size(); i++) {
                StreamConfig in_stream = sub_config.get_input_streams()[i];
                for (NodeConfig &node:sub_config.nodes) {
                    for (int j = 0; j < node.get_input_streams().size(); j++) {
                        StreamConfig node_in_stream = node.get_input_streams()[j];
                        if (in_stream.get_identifier() == node_in_stream.get_identifier()) {
                            node.input_streams[j] = sub_graph_node.input_streams[i];
                            break;
                        }
                    }
                }
            }

            // find tail nodes from subgraph-inside-nodes and replace their output_streams
            for (int i = 0; i < sub_config.get_output_streams().size(); i++) {
                StreamConfig out_stream = sub_config.get_output_streams()[i];
                for (NodeConfig &node:sub_config.nodes) {
                    for (int j = 0; j < node.get_output_streams().size(); j++) {
                        StreamConfig node_out_stream = node.get_output_streams()[j];
                        if (out_stream.get_identifier() == node_out_stream.get_identifier()) {
                            node.output_streams[j] = sub_graph_node.output_streams[i];
                            break;
                        }
                    }
                }
            }

            // remove subgraph node from main graph
            main_config.nodes.erase(std::remove(main_config.nodes.begin(), main_config.nodes.end(), sub_graph_node),
                                    main_config.nodes.end());

            // add all subgraph-inside-nodes into main graph
            for (NodeConfig &node:sub_config.nodes) {
                main_config.nodes.push_back(node);
            }
        }

        void subgraph_preprocess(GraphConfig &main_graph_config, std::map<int, std::shared_ptr<Module> > &created_modules) {
            GraphConfig main_graph_config_tmp = main_graph_config;

            for (NodeConfig &node:main_graph_config.nodes) {
                // skip pre-module
                if (node.get_node_meta().get_premodule_id() != -1) {
                    continue;
                }

                int node_id = node.get_id();
                std::string module_info = node.get_module_info().to_json().dump();
                std::string module_opt = node.get_option().dump();
                // judge subgraph
                std::pair<bool, std::shared_ptr<Module> > subgraph_check;
                if (created_modules.count(node_id))
                    subgraph_check = {bmf_engine::ModuleFactory::test_subgraph(created_modules[node_id]),
                                      created_modules[node_id]};
                else
                    subgraph_check = bmf_engine::ModuleFactory::create_and_test_subgraph(module_info, node_id,
                                                                                         module_opt);

                if (subgraph_check.first) {
                    // get subgraph config
                    JsonParam subgraph_config_ = bmf_engine::ModuleFactory::get_subgraph_config(subgraph_check.second);
                    json subgraph_json = subgraph_config_.json_value_;
                    GraphConfig subgraph_config = bmf_engine::GraphConfig(subgraph_json);
                    subgraph_preprocess(subgraph_config, created_modules);

                    // Merge sub-graph with main graph
                    merge_subgraph(main_graph_config_tmp, subgraph_config, node_id);
                } else
                    created_modules[node_id] = subgraph_check.second;
            }
            main_graph_config = main_graph_config_tmp;
        }

        void dump_graph(GraphConfig graph_config, bool merged) {
            json option = graph_config.get_option().json_value_;

            // need dump
            if (option.find("dump_graph") != option.end() && option["dump_graph"] == 1) {
                std::string graph_config_info = graph_config.to_json().dump(4, ' ');

                std::string file_name;

                // file name
                if (option.find("graph_name") == option.end()) {
                    if (!merged) {
                        file_name = "graph_unmerged.json";
                    } else {
                        file_name = "graph.json";
                    }
                } else {
                    if (!merged) {
                        file_name = option["graph_name"].get<std::string>() + "_unmerged.json";
                    } else {
                        file_name = option["graph_name"].get<std::string>() + ".json";
                    }
                }

                // write graph info into file
                std::ofstream fout;
                fout.open(file_name);
                fout << graph_config_info << std::endl;
                fout.close();
            }
        }
    }

END_BMF_ENGINE_NS
