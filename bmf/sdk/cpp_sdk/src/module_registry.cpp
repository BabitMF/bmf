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

#include <bmf/sdk/module_registry.h>

BEGIN_BMF_SDK_NS
    ModuleRegistry::ConstructorRegistry &ModuleRegistry::Registry() {
        static ConstructorRegistry *real_registry = new ConstructorRegistry();
        return *real_registry;
    }

    void ModuleRegistry::AddConstructor(std::string const &module_name, std::string const &sdk_version,
                                        Constructor constructor) {
        ConstructorRegistry &registry = Registry();
        registry[module_name] = {sdk_version, constructor};
    }

    std::shared_ptr<Module>
    ModuleRegistry::ConstructModule(std::string const &module_name, int node_id, JsonParam json_param) {
        ConstructorRegistry &registry = Registry();
        return registry[module_name].second(node_id, json_param);
    }

    std::string ModuleRegistry::GetModuleUsingSDKVersion(const std::string &module_name) {
        ConstructorRegistry &registry = Registry();
        return registry[module_name].first;
    }

    ModuleRegister::ModuleRegister(std::string const &module_name, std::string const &sdk_version,
                                   std::shared_ptr<Module> (*constructor)(int node_id, JsonParam json_param)) {
        ModuleRegistry::AddConstructor(module_name, sdk_version, constructor);
    }

    ModuleRegister::ModuleRegister(const std::string &module_name,
                                   std::shared_ptr<Module> (*constructor)(int, JsonParam)) {
        ModuleRegistry::AddConstructor(module_name, "V0.0.1", constructor);
    }
END_BMF_SDK_NS