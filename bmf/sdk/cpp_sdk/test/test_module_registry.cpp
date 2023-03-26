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
//#include <bmf/sdk/../../../c_modules/include/c_module.h>
#include <gtest/gtest.h>


USE_BMF_SDK_NS

class CFFTestmodule : public Module {
    public:
        CFFTestmodule(int node_id, JsonParam option)
            : Module(node_id,option) {}
        int32_t process(Task &task) {return 0;}
};

REGISTER_MODULE_CLASS(CFFTestmodule)

TEST(module_registry, construct_module) {
    std::string module_name = "CFFTestmodule";
    int node_id = 1;
    std::string option_str = "{\"name\":\"mock_test_module\"}";
    JsonParam json_param = JsonParam(option_str);
    std::shared_ptr<Module> test_module = ModuleRegistry::ConstructModule(module_name, node_id, json_param);
    std::string sdk_version = ModuleRegistry::GetModuleUsingSDKVersion(module_name);
    EXPECT_EQ(sdk_version, BMF_SDK_VERSION);
}

TEST(module_registry, init_by_module_name) {
    std::string module_name = "CFFTestmodule";
    int node_id = 1;
    std::string option_str = "{\"name\":\"mock_test_module\"}";
    JsonParam json_param = JsonParam(option_str);
    ModuleRegister module_registry = ModuleRegister(module_name, nullptr);
    std::string sdk_version = ModuleRegistry::GetModuleUsingSDKVersion(module_name);
    EXPECT_EQ(sdk_version, "V0.0.1");
}

TEST(module_registry, init_by_module_name_and_version) {
    std::string module_name = "CFFTestmodule";
    std::string version = "V0.0.3";
    int node_id = 1;
    std::string option_str = "{\"name\":\"mock_test_module\"}";
    JsonParam json_param = JsonParam(option_str);
    ModuleRegister module_registry = ModuleRegister(module_name, version, nullptr);
    std::string sdk_version = ModuleRegistry::GetModuleUsingSDKVersion(module_name);
    EXPECT_EQ(sdk_version, "V0.0.3");
}

