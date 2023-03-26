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

#include "../include/graph_config.h"

#include <bmf/sdk/log.h>

#include <bmf_nlohmann/json_fwd.hpp>

#include <iostream>
#include <string>
#include <list>
#include <fstream>
#include <sstream>
#include <unordered_set>
#include <unordered_map>

BEGIN_BMF_ENGINE_NS
    USE_BMF_SDK_NS

    StreamConfig::StreamConfig(JsonParam &stream_config) {
        init(stream_config.json_value_);
    }

    StreamConfig::StreamConfig(bmf_nlohmann::json &stream_config) {
        init(stream_config);
    }

    void StreamConfig::init(bmf_nlohmann::json &stream_config) {
        identifier = stream_config.at("identifier").get<std::string>();
        auto p = identifier.find(':');
        if (p != std::string::npos) {
            notify = identifier.substr(0, p);
            identifier = identifier.substr(p + 1, identifier.length() - p);
        } else {
            notify = "";
        }
        if (stream_config.count("alias"))
            alias = stream_config.at("alias").get<std::string>();
    }

    std::string StreamConfig::get_identifier() {
        return identifier;
    }

    std::string StreamConfig::get_notify() {
        return notify;
    }

    std::string StreamConfig::get_alias() {
        return alias;
    }

    bmf_nlohmann::json StreamConfig::to_json(){
        bmf_nlohmann::json json_stream_info;
        json_stream_info["identifier"] = identifier;
        json_stream_info["notify"] = notify;
        json_stream_info["alias"] = alias;
        return json_stream_info;
    }

    ModuleConfig::ModuleConfig(JsonParam &module_config) {
        init(module_config.json_value_);
    }

    ModuleConfig::ModuleConfig(bmf_nlohmann::json &module_config) {
        init(module_config);
    }

    void ModuleConfig::init(bmf_nlohmann::json &module_config) {
        if (module_config.count("name"))
            module_name = module_config.at("name").get<std::string>();
        if (module_config.count("type"))
            module_type = module_config.at("type").get<std::string>();
        if (module_config.count("path"))
            module_path = module_config.at("path").get<std::string>();
        if (module_config.count("entry"))
            module_entry = module_config.at("entry").get<std::string>();
    }

    std::string ModuleConfig::get_module_name(){
        return module_name;
    }

    std::string ModuleConfig::get_module_type(){
        return module_type;
    }

    std::string ModuleConfig::get_module_path(){
        return module_path;
    }

    std::string ModuleConfig::get_module_entry(){
        return module_entry;
    }

    bmf_nlohmann::json ModuleConfig::to_json(){
        bmf_nlohmann::json json_module_info;
        json_module_info["name"] = module_name;
        json_module_info["type"] = module_type;
        json_module_info["path"] = module_path;
        json_module_info["entry"] = module_entry;
        return json_module_info;
    }

    NodeMetaInfo::NodeMetaInfo(JsonParam &node_meta) {
        init(node_meta.json_value_);
    }

    NodeMetaInfo::NodeMetaInfo(bmf_nlohmann::json &node_meta) {
        init(node_meta);
    }

    void NodeMetaInfo::init(bmf_nlohmann::json &node_meta) {
        if (node_meta.count("bundle_id"))
            bundle = node_meta.at("bundle_id").get<int32_t>();
        if (node_meta.count("premodule_id"))
            premodule_id = node_meta.at("premodule_id").get<int32_t>();
        if (node_meta.count("callback_binding"))
            for (auto &cb:node_meta.at("callback_binding")) {
                auto binding = cb.get<std::string>();
                auto p = binding.find(':');
                if (p == std::string::npos)
                    throw std::logic_error("Wrong callback binding.");
                callback_binding[std::stoll(binding.substr(0, p))] =
                        std::stoul(binding.substr(p + 1, binding.length() - p));
            }
        if (node_meta.count("queue_length_limit"))
            queue_size_limit = node_meta.at("queue_length_limit").get<uint32_t>();
    }

    int32_t NodeMetaInfo::get_premodule_id() {
        return premodule_id;
    }

    int32_t NodeMetaInfo::get_bundle() {
        return bundle;
    }

    uint32_t NodeMetaInfo::get_queue_size_limit() {
        return queue_size_limit;
    }

    std::map<int64_t, uint32_t> NodeMetaInfo::get_callback_binding() {
        return callback_binding;
    }

    bmf_nlohmann::json NodeMetaInfo::to_json(){
        bmf_nlohmann::json json_meta_info;
        json_meta_info["premodule_id"] = premodule_id;
        json_meta_info["callback_binding"] = bmf_nlohmann::json(std::vector<std::string>());
        for (auto &it:callback_binding){
            json_meta_info["callback_binding"].push_back(std::to_string(it.first)+":"+std::to_string(it.second));
        }
        return json_meta_info;
    }

    NodeConfig::NodeConfig(JsonParam &node_config) {
        init(node_config.json_value_);
    }

    NodeConfig::NodeConfig(bmf_nlohmann::json &node_config) {
        init(node_config);
    }

    std::string NodeConfig::get_alias() {
        return alias;
    }

    std::string NodeConfig::get_action() {
        return action;
    }

    void NodeConfig::init(bmf_nlohmann::json &node_config) {
        if (node_config.count("id"))
            id = node_config.at("id").get<int>();

        if (node_config.count("module_info"))
            module = ModuleConfig(node_config.at("module_info"));

        if (node_config.count("meta_info"))
            meta = NodeMetaInfo(node_config.at("meta_info"));

        if (node_config.count("input_streams"))
            for (auto s:node_config.at("input_streams"))
                input_streams.emplace_back(s);
        if (node_config.count("output_streams"))
            for (auto s:node_config.at("output_streams"))
                output_streams.emplace_back(s);
        if (node_config.count("option"))
            option = JsonParam(node_config.at("option"));
        if (node_config.count("scheduler"))
            scheduler = node_config.at("scheduler").get<int>();
        if (node_config.count("input_manager"))
            input_manager = node_config.at("input_manager").get<std::string>();
        if (node_config.count("alias"))
            alias = node_config.at("alias").get<std::string>();
        if (node_config.count("action"))
            action = node_config.at("action").get<std::string>();
    }

    ModuleConfig NodeConfig::get_module_info() {
        return module;
    }

    JsonParam NodeConfig::get_option() {
        return option;
    }

    void NodeConfig::set_option(JsonParam node_option){
        option = node_option;
    }

    std::string NodeConfig::get_input_manager() {
        return input_manager;
    }

    NodeMetaInfo NodeConfig::get_node_meta() {
        return meta;
    }

    std::vector<StreamConfig>& NodeConfig::get_input_streams() {
        return input_streams;
    }

    std::vector<StreamConfig>& NodeConfig::get_output_streams() {
        return output_streams;
    }

    void NodeConfig::add_input_stream(StreamConfig input_stream){
        input_streams.push_back(input_stream);
    }

    void NodeConfig::add_output_stream(StreamConfig output_stream){
        output_streams.push_back(output_stream);
    }

    int NodeConfig::get_id() {
        return id;
    }

    int NodeConfig::get_scheduler() {
        return scheduler;
    }

    bmf_nlohmann::json NodeConfig::to_json() {
        bmf_nlohmann::json json_node_config;
        json_node_config["id"] = id;
        json_node_config["scheduler"] = scheduler;
        json_node_config["input_manager"] = input_manager;
        json_node_config["module_info"] = module.to_json();
        json_node_config["meta_info"] = meta.to_json();
        json_node_config["option"] = option.json_value_;

        json_node_config["input_streams"] = bmf_nlohmann::json(std::vector<bmf_nlohmann::json>());
        for (StreamConfig input_stream:input_streams){
            json_node_config["input_streams"].push_back(input_stream.to_json());
        }

        json_node_config["output_streams"] = bmf_nlohmann::json(std::vector<bmf_nlohmann::json>());
        for (StreamConfig output_stream:output_streams){
            json_node_config["output_streams"].push_back(output_stream.to_json());
        }

        return json_node_config;
    }

    GraphConfig::GraphConfig(std::string config_file) {
        if (config_file == "")
            throw std::logic_error("No config file for graph config!");

        bmf_nlohmann::json graph_json;
        std::ifstream gs(config_file);
        gs >> graph_json;
        init(graph_json);
    }

    GraphConfig::GraphConfig(JsonParam &graph_config) {
        init(graph_config.json_value_);
    }

    GraphConfig::GraphConfig(bmf_nlohmann::json &graph_config) {
        init(graph_config);
    }

    void GraphConfig::init(bmf_nlohmann::json &graph_config) {
        auto validate = [](bmf_nlohmann::json graph) -> void {
            std::unordered_map<std::string, std::string> input_streams;
            std::unordered_map<std::string, std::string> output_streams;
            std::unordered_map<std::string, std::unordered_set<std::string> > edges;

            auto unwrap_identifier = [](std::string const &identifier) -> std::string {
                auto p = identifier.find(':');
                if (p != std::string::npos)
                    return identifier.substr(p + 1, identifier.length() - p);
                return identifier;
            };

            auto validation_insert_output_stream = [&unwrap_identifier](std::string const &module,
                                                                        std::string const &stream,
                                                                        std::unordered_map<std::string, std::string> &stream_set) -> void {
                auto s = unwrap_identifier(stream);
                if (stream_set.count(s))
                    throw std::logic_error("Duplicated input_stream in graph!");
                stream_set[s] = module;
            };

            auto validation_decoder_output = [](std::string const &stream) -> void {
                auto p = stream.find(':');
                if (p != std::string::npos){
                    std::string stream_name = stream.substr(0, p);
                    if (!(stream_name == "video" || stream_name == "audio"))
                        throw std::logic_error("Incorrect stream notify for decoder!");
                }
            };

            auto validation_insert_edge = [&unwrap_identifier, &edges](std::string const &in_s,
                                                                       std::string const &out_s) -> void {
                auto s1 = unwrap_identifier(in_s), s2 = unwrap_identifier(out_s);
                if (!edges.count(s1))
                    edges[s1] = std::unordered_set<std::string>();
                if (edges[s1].count(s2))
                    throw std::logic_error("Duplicated edge in graph!");
                edges[s1].insert(s2);
            };

            // Graph mode is necessary.
            std::string run_mode = "";
            if (graph.count("mode"))
                run_mode = graph.at("mode").get<std::string>();

            // In Server/Generator mode, graph must have input_stream/output_stream;
            // Otherwise, graph may have some, but validator will ignore them.
            if (run_mode == "Server" || run_mode == "Generator" || run_mode == "Pushdata") {
                for (auto stream:graph.at("input_streams"))
                    validation_insert_output_stream("BMF_SUPER_SOURCE", stream.at("identifier").get<std::string>(),
                                                    input_streams);
                for (auto stream:graph.at("output_streams"))
                    edges[stream.at("identifier").get<std::string>()] = std::unordered_set<std::string>();
                if (run_mode == "Server" && input_streams.empty())
                    throw std::logic_error("Server Mode require input_streams of graph to execute.");
                if (edges.empty() && run_mode != "Pushdata")
                    throw std::logic_error("Server/Generator Mode require output_streams of graph to execute.");
            }

            //--------------------------------VALIDATION-----------------------------------------
            // Collect info and check some basic syntax errors.
            for (auto node:graph.at("nodes")) {
                // Check whether "id" exists.
                if (!node.count("action") && !node.count("id"))
                    throw std::logic_error("Missing 'id' parameter.");
                // Check whether "module" exists.
                if ((!node.count("action")
                        || (node.count("action") && node.at("action").get<std::string>() == "add"))
                    && !node.count("module_info"))
                    throw std::logic_error("Missing 'module_info' parameter.");
                // Collect stream info.
                std::string name = "";
                if (node.count("module_info"))
                    name = node.at("module_info").at("name").get<std::string>() + "_" +
                                std::to_string(node.at("id").get<int>());
                if (node.count("output_streams") && node.at("output_streams").empty()) {}
                    //for (auto in_s:node.at("input_streams"))
                    //    validation_insert_edge(in_s.at("identifier").get<std::string>(), "BMF_SUPER_DESTINATION");
                else {
                    // Check output for decoder
                    if (node.count("module_info") && node.at("module_info").at("name").get<std::string>() == "c_ffmpeg_decoder") {
                        if (node.count("output_streams"))
                            for (auto out_s:node.at("output_streams")) {
                                validation_decoder_output(out_s.at("identifier").get<std::string>());
                            }
                    }
                    if (node.count("output_streams"))
                        for (auto out_s:node.at("output_streams")) {
                            validation_insert_output_stream(name, out_s.at("identifier").get<std::string>(),
                                                            output_streams);
                            for (auto in_s:node.at("input_streams"))
                                validation_insert_edge(in_s.at("identifier").get<std::string>(),
                                                       out_s.at("identifier").get<std::string>());
                        }
                }
            }
// TODO: Possibility of input_stream existing as a placeholder, ignore this check temporally.
//            // Check whether in/out pin matches.
//            for (auto &edge:edges)
//                if (!output_streams.count(edge.first)) {
//                    throw std::logic_error("Using unexisting input_stream.");
//                }
            // Check looping, using DFS.
            std::unordered_map<std::string, bool> vis;
            std::unordered_set<std::string> checked;
            std::function<bool(std::string)> dfs;
            dfs = [&vis, &edges, &dfs](const std::string &in_s) -> bool {
                if (vis.count(in_s) && vis[in_s])
                    return true;
                vis[in_s] = true;
                for (auto &out_s:edges[in_s])
                    if (dfs(out_s))
                        return true;
                vis[in_s] = false;
                return false;
            };
            for (auto &edge:edges)
                if (!checked.count(input_streams[edge.first]) && dfs(edge.first))
                    throw std::logic_error("Loop in graph!");
                else
                    checked.insert(input_streams[edge.first]);
            // TODO: Server/Generator mode need extra checking?
        };

        validate(graph_config);

        mode = BmfMode::NORMAL_MODE;
        if (graph_config.count("mode"))
            mode = [](std::string mod) -> BmfMode {
                if (mod == "Server")
                    return BmfMode::SERVER_MODE;
                if (mod == "Generator")
                    return BmfMode::GENERATOR_MODE;
                if (mod == "Subgraph")
                    return BmfMode::SUBGRAPH_MODE;
                if (mod == "Pushdata")
                    return BmfMode::PUSHDATA_MODE;
                return BmfMode::NORMAL_MODE;
            }(graph_config.at("mode").get<std::string>());

        // Get option if needed.
        if (graph_config.count("option"))
            option = JsonParam(graph_config.at("option"));

        if (!graph_config.count("nodes"))
            throw std::logic_error("Missing nodes in graph config.");
        for (auto nd:graph_config.at("nodes"))
            nodes.emplace_back(NodeConfig(nd));
        if (mode == BmfMode::SERVER_MODE || mode == BmfMode::GENERATOR_MODE || mode == BmfMode::SUBGRAPH_MODE) {
            if (!graph_config.count("input_streams"))
                throw std::logic_error("Missing input_steams in graph_config at server/generator mode.");
            for (auto s:graph_config.at("input_streams"))
                input_streams.emplace_back(StreamConfig(s));
            if (!graph_config.count("output_streams"))
                throw std::logic_error("Missing output_steams in graph_config at server/generator mode.");
            for (auto s:graph_config.at("output_streams"))
                output_streams.emplace_back(StreamConfig(s));
        }
        if (mode == BmfMode::PUSHDATA_MODE) {
            if (!graph_config.count("input_streams"))
                throw std::logic_error("Missing input_steams in graph_config at push data mode.");
            for (auto s:graph_config.at("input_streams"))
                input_streams.emplace_back(StreamConfig(s));
        }
    }

    JsonParam GraphConfig::get_option() {
        return option;
    }

    BmfMode GraphConfig::get_mode() {
        return mode;
    }

    std::vector<NodeConfig> GraphConfig::get_nodes() {
        return nodes;
    }

    std::vector<StreamConfig> GraphConfig::get_input_streams() {
        return input_streams;
    }

    std::vector<StreamConfig> GraphConfig::get_output_streams() {
        return output_streams;
    }

    bmf_nlohmann::json GraphConfig::to_json(){
        bmf_nlohmann::json json_graph_config;
        json_graph_config["option"] = option.json_value_;

        if (mode == BmfMode::NORMAL_MODE){
            json_graph_config["mode"] = "normal";
        }else if (mode == BmfMode::SERVER_MODE){
            json_graph_config["mode"] = "server";
        }else if (mode == BmfMode::GENERATOR_MODE){
            json_graph_config["mode"] = "generator";
        }else if (mode == BmfMode::SUBGRAPH_MODE){
            json_graph_config["mode"] = "subgraph";
        }else if (mode == BmfMode::PUSHDATA_MODE){
            json_graph_config["mode"] = "pushdata";
        }

        json_graph_config["input_streams"] = bmf_nlohmann::json(std::vector<bmf_nlohmann::json>());

        for (StreamConfig input_stream:input_streams){
            json_graph_config["input_streams"].push_back(input_stream.to_json());
        }

        json_graph_config["output_streams"] = bmf_nlohmann::json(std::vector<bmf_nlohmann::json>());
        for (StreamConfig output_stream:output_streams){
            json_graph_config["output_streams"].push_back(output_stream.to_json());
        }

        json_graph_config["nodes"] = bmf_nlohmann::json(std::vector<bmf_nlohmann::json>());
        for (NodeConfig node_config:nodes){
            json_graph_config["nodes"].push_back(node_config.to_json());
        }

        return json_graph_config;
    }

END_BMF_ENGINE_NS
