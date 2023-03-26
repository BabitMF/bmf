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

#include "../include/connector.hpp"

#include "../internal/env_init.h"
#include "../internal/mapping.h"

#include "../../c_engine/include/callback_layer.h"
#include "../../c_engine/include/graph_config.h"
#include "../../c_engine/include/graph.h"
#include "../../c_engine/include/module_factory.h"
#include "../../c_engine/include/optimizer.h"

#include <bmf/sdk/cbytes.h>
#include <bmf/sdk/json_param.h>
#include <bmf/sdk/compat/path.h>

#include <bmf_nlohmann/json.hpp>

#include <functional>
#include <memory>

using json_t = bmf_nlohmann::json;

namespace bmf {
    BMFGraph::BMFGraph(const std::string &graph_config, bool is_path, bool need_merge) {
        
        json_t graph_json;
        if (is_path) {
            if (!fs::exists(graph_config))
                throw std::logic_error("Graph config file not exists.");
            std::ifstream gs(graph_config);
            gs >> graph_json;
        } else {
            graph_json = json_t::parse(graph_config);
        }

        auto g_config = bmf_engine::GraphConfig(graph_json);
        std::map<int, std::shared_ptr<bmf_sdk::Module> > created_modules;
        std::map<int, std::shared_ptr<bmf_engine::ModuleCallbackLayer> > callback_bindings;

        // find subgraph and merge into while graph
        bmf_engine::Optimizer::subgraph_preprocess(g_config, created_modules);

        // catch graph config at this time for further comparison.
        auto ori_g_config = g_config;

        // dump unmerged graph
        bmf_engine::Optimizer::dump_graph(g_config, false);

        // convert filter para to new format
        bmf_engine::Optimizer::convert_filter_para_for_graph(g_config.nodes);

        // do optimize for filter nodes
        if (need_merge) {
            bmf_engine::Optimizer::optimize(g_config.nodes);
        }

        // replace stream name with stream id in filter option
        bmf_engine::Optimizer::replace_stream_name_for_graph(g_config.nodes);

        // dump merged graph
        bmf_engine::Optimizer::dump_graph(g_config, true);

        for (auto &nd:g_config.nodes) {
            auto meta = nd.meta;
            callback_bindings[nd.id] = std::make_shared<bmf_engine::ModuleCallbackLayer>();
            if (meta.get_premodule_id() > 0) {
                if (!internal::ConnectorMapping::ModuleInstanceMapping().exist(meta.get_premodule_id()))
                    throw std::logic_error("Trying to use an unexisted premodule.");
                created_modules[nd.id] = internal::ConnectorMapping::ModuleInstanceMapping().get(meta.get_premodule_id());
            }

            for (auto cb: meta.get_callback_binding()) {
                if (!internal::ConnectorMapping::ModuleCallbackInstanceMapping().exist(cb.second))
                    throw std::logic_error("Trying to bind an unexisted callback.");
                callback_bindings[nd.id]->add_callback(
                        cb.first, *internal::ConnectorMapping::ModuleCallbackInstanceMapping().get(cb.second));
            }
        }

        // Clean up pre-modules.
        std::map<int, std::shared_ptr<bmf_sdk::Module> > all_created_modules;
        std::map<int, std::string> node_ori_opts;
        for (auto nd:ori_g_config.get_nodes())
            node_ori_opts[nd.id] = nd.option.dump();
        for (auto nd:g_config.get_nodes()) {
            if (node_ori_opts.count(nd.id) && node_ori_opts[nd.id] != nd.option.dump())
                continue;
            if (created_modules.count(nd.id))
                all_created_modules[nd.id] = created_modules[nd.id];
        }
        created_modules.clear();

        auto g_instance = std::make_shared<bmf_engine::Graph>(g_config, all_created_modules, callback_bindings);

        graph_uid_ = internal::ConnectorMapping::GraphInstanceMapping().insert(g_instance);
    }

    BMFGraph::BMFGraph(BMFGraph const &graph) {
        graph_uid_ = graph.graph_uid_;
        internal::ConnectorMapping::GraphInstanceMapping().ref(graph_uid_);
    }

    BMFGraph::~BMFGraph() {
        internal::ConnectorMapping::GraphInstanceMapping().remove(graph_uid_);
    }

    BMFGraph &BMFGraph::operator=(BMFGraph const &graph) {
        internal::ConnectorMapping::GraphInstanceMapping().ref(graph.graph_uid_);
        internal::ConnectorMapping::GraphInstanceMapping().remove(graph_uid_);
        graph_uid_ = graph.graph_uid_;

        return *this;
    }

    uint32_t BMFGraph::uid() {
        return graph_uid_;
    }

    void BMFGraph::start() {
        internal::ConnectorMapping::GraphInstanceMapping().get(graph_uid_)->start();
    }

    void BMFGraph::update(const std::string &graph_config, bool is_path) {
        json_t graph_json;
        if (is_path) {
            if (!fs::exists(graph_config))
                throw std::logic_error("Graph config file not exists.");
            std::ifstream gs(graph_config);
            gs >> graph_json;
        } else {
            graph_json = json_t::parse(graph_config);
        }
        auto g_config = bmf_engine::GraphConfig(graph_json);
        
        internal::ConnectorMapping::GraphInstanceMapping().get(graph_uid_)->update(g_config);
    }

    int BMFGraph::close() {
        return internal::ConnectorMapping::GraphInstanceMapping().get(graph_uid_)->close();
    }

    int BMFGraph::force_close() {
        return internal::ConnectorMapping::GraphInstanceMapping().get(graph_uid_)->force_close();
    }

    int BMFGraph::add_input_stream_packet(const std::string &stream_name, bmf_sdk::Packet &packet, bool block) {
        return internal::ConnectorMapping::GraphInstanceMapping().get(graph_uid_)->add_input_stream_packet(stream_name,
                                                                                                           packet, block);
    }

    bmf_sdk::Packet BMFGraph::poll_output_stream_packet(const std::string &stream_name, bool block) {
        return internal::ConnectorMapping::GraphInstanceMapping().get(graph_uid_)->poll_output_stream_packet(
                stream_name, block);
    }

    GraphRunningInfo BMFGraph::status() {
        return internal::ConnectorMapping::GraphInstanceMapping().get(graph_uid_)->status();
    }

    BMFModule::BMFModule(const std::string &module_name, const std::string &option, const std::string &module_type,
                         const std::string &module_path, const std::string &module_entry) {
        std::shared_ptr<bmf_sdk::Module> mod;
        auto opt = bmf_sdk::JsonParam(option);
        module_name_ = module_name;
        auto m = bmf_engine::ModuleFactory::create_module(module_name, -1, opt, module_type, module_path, module_entry,
                                                          mod);
        if(!mod){
            throw std::runtime_error("Load module " + module_name + " failed");
        }
        auto cb = std::make_shared<bmf_engine::ModuleCallbackLayer>();
        mod->set_callback([cb](int64_t key, CBytes para) -> CBytes {
            return cb->call(key, para);
        });
        module_uid_ = internal::ConnectorMapping::ModuleInstanceMapping().insert(mod);
    }

    BMFModule::BMFModule(std::shared_ptr<bmf_sdk::Module> module_p) {
        module_uid_ = internal::ConnectorMapping::ModuleInstanceMapping().insert(module_p);
    }

    BMFModule::BMFModule(BMFModule const &mod) {
        module_uid_ = mod.module_uid_;
        internal::ConnectorMapping::ModuleInstanceMapping().ref(module_uid_);
    }

    BMFModule::~BMFModule() {
        internal::ConnectorMapping::ModuleInstanceMapping().remove(module_uid_);
    }

    BMFModule &BMFModule::operator=(BMFModule const &mod) {
        internal::ConnectorMapping::ModuleInstanceMapping().ref(mod.module_uid_);
        internal::ConnectorMapping::ModuleInstanceMapping().remove(module_uid_);
        module_uid_ = mod.module_uid_;

        return *this;
    }

    uint32_t BMFModule::uid() {
        return module_uid_;
    }

    int32_t BMFModule::process(bmf_sdk::Task &task) {
        BMF_TRACE(PROCESSING, module_name_.c_str(), START);
        int32_t status = internal::ConnectorMapping::ModuleInstanceMapping().get(module_uid_)->process(task);
        BMF_TRACE(PROCESSING, module_name_.c_str(), END);
        return status;
    }

    int32_t BMFModule::init() {
        return internal::ConnectorMapping::ModuleInstanceMapping().get(module_uid_)->init();
    }

    int32_t BMFModule::reset() {
        return internal::ConnectorMapping::ModuleInstanceMapping().get(module_uid_)->reset();
    }

    int32_t BMFModule::close() {
        return internal::ConnectorMapping::ModuleInstanceMapping().get(module_uid_)->close();
    }

    BMFCallback::BMFCallback(std::function<bmf_sdk::CBytes(bmf_sdk::CBytes)> callback) {
        callback_uid_ = internal::ConnectorMapping::ModuleCallbackInstanceMapping().insert(
                std::make_shared<std::function<bmf_sdk::CBytes(bmf_sdk::CBytes)> >(callback));
    }

    BMFCallback::BMFCallback(BMFCallback const &cb) {
        callback_uid_ = cb.callback_uid_;
        internal::ConnectorMapping::ModuleCallbackInstanceMapping().ref(callback_uid_);
    }

    BMFCallback::~BMFCallback() {
        internal::ConnectorMapping::ModuleCallbackInstanceMapping().remove(callback_uid_);
    }

    BMFCallback &BMFCallback::operator=(BMFCallback const &cb) {
        internal::ConnectorMapping::ModuleCallbackInstanceMapping().ref(cb.callback_uid_);
        internal::ConnectorMapping::ModuleCallbackInstanceMapping().remove(callback_uid_);
        callback_uid_ = cb.callback_uid_;

        return *this;
    }

    uint32_t BMFCallback::uid() {
        return callback_uid_;
    }

    void ChangeDmpPath(std::string path) {
        internal::env_init.ChangeDmpPath(path);
    }
}
