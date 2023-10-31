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
#include "../include/module_factory.h"

#include "gtest/gtest.h"
#include <fstream>

#ifndef BMF_ENABLE_MOBILE

#include <filesystem>

namespace fs = std::filesystem;

USE_BMF_ENGINE_NS
USE_BMF_SDK_NS
TEST(module_factory, cpp_module) {
    std::string module_name = "cpp_copy_module";
    int node_id = 1;
    JsonParam option;
    std::string module_type;
    std::string module_path;
    std::string module_entry;
    std::shared_ptr<Module> module;
    ModuleInfo module_info =
        ModuleFactory::create_module(module_name, node_id, option, module_type,
                                     module_path, module_entry, module);
    EXPECT_EQ(module == nullptr, 0);
}

TEST(module_factory, python_module) {
    std::string module_name = "python_copy_module";
    int node_id = 0;
    JsonParam option;
    std::string option_str = "{\"hello\":1}";
    option.parse(option_str);
    std::string module_type;
    std::string module_path;
    std::string module_entry;
    std::shared_ptr<Module> module;
    ModuleInfo module_info =
        ModuleFactory::create_module(module_name, node_id, option, module_type,
                                     module_path, module_entry, module);
    EXPECT_EQ(module == nullptr, 0);
}

#endif // BMF_ENABLE_MOBILE
