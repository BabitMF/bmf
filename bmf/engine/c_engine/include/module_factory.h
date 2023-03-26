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

#ifndef BMF_MODULE_FACTORY_H
#define BMF_MODULE_FACTORY_H

#include <bmf/sdk/module_manager.h>
#include <bmf/sdk/json_param.h>
#include <bmf/sdk/module.h>

#include <unordered_map>
#include <unordered_set>

BEGIN_BMF_ENGINE_NS
    USE_BMF_SDK_NS

    class ModuleFactory {
    public:
        static ModuleInfo create_module(std::string module_name, int node_id, JsonParam &option,
                                        std::string module_type, std::string module_path, std::string module_entry,
                                        std::shared_ptr<Module> &module);

        static JsonParam
        get_subgraph_config(std::shared_ptr<Module> module_instance);

        static JsonParam
        create_and_get_subgraph_config(std::string module_info, int node_id, std::string option);

        static std::pair<bool, std::shared_ptr<Module> >
        create_and_test_subgraph(std::string module_info, int node_id, std::string option);

        static bool test_subgraph(std::shared_ptr<Module> module_instance);
    };

END_BMF_ENGINE_NS

#endif //BMF_MODULE_FACTORY_H
