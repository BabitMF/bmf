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

#include "../include/module_factory.h"
#include "../include/graph_config.h"

#include <bmf/sdk/module_manager.h>
#include <bmf/sdk/module_registry.h>

#include <bmf_nlohmann/json_fwd.hpp>

#include <dlfcn.h>
#include <memory>
#include <unordered_set>


BEGIN_BMF_ENGINE_NS
    USE_BMF_SDK_NS

    ModuleInfo ModuleFactory::create_module(std::string module_name, int node_id, JsonParam &option,
                                            std::string module_type, std::string module_path, std::string module_entry,
                                            std::shared_ptr<Module> &module) {
        auto &M = bmf_sdk::ModuleManager::instance();

        ModuleInfo module_info;
        auto factory = M.load_module(module_name, module_type, module_path, module_entry, &module_info);
        if(factory == nullptr){
            throw std::runtime_error(fmt::format("create module {} failed", module_name));
        }
        module = factory->make(node_id, option);
        return module_info;
    }

    JsonParam
    ModuleFactory::get_subgraph_config(std::shared_ptr<Module> module_instance) {
        JsonParam graph_config;
        module_instance->get_graph_config(graph_config);
        return graph_config;
    }

    JsonParam
    ModuleFactory::create_and_get_subgraph_config(std::string module_info, int node_id, std::string option) {
        auto tmp = bmf_nlohmann::json::parse(module_info);
        auto info = ModuleConfig(tmp);
        auto opt = JsonParam(bmf_nlohmann::json::parse(option));
        std::shared_ptr<Module> module;
        create_module(info.module_name, node_id, opt, info.module_type, info.module_path, info.module_entry, module);
        // get graph_config of subgraph
        JsonParam graph_config;
        module->get_graph_config(graph_config);
        return graph_config;
    }

    bool ModuleFactory::test_subgraph(std::shared_ptr<Module> module_instance) {
        return module_instance->is_subgraph();
    }

    std::pair<bool, std::shared_ptr<Module>>
    ModuleFactory::create_and_test_subgraph(std::string module_info, int node_id, std::string option) {
        auto tmp = bmf_nlohmann::json::parse(module_info);
        auto info = ModuleConfig(tmp);
        auto opt = JsonParam(bmf_nlohmann::json::parse(option));
        std::shared_ptr<Module> module_instance;
        create_module(info.module_name, node_id, opt, info.module_type, info.module_path, info.module_entry,
                      module_instance);
        return {module_instance->is_subgraph(), module_instance};
    }

END_BMF_ENGINE_NS
