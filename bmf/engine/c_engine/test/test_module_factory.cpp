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
    std::string module_name = "test_cpp_module";
    std::string dst_dir = "/opt/tiger/bmf_mods/Module_"+module_name+"/";
    std::string dst_so_file = dst_dir + "libcopy_module.so";
    std::string dst_meta_file = dst_dir + "meta.info";
    std::string source_dir = "../../../output/example/c_module";
    std::string source_so_file = source_dir+"/libcopy_module.so";
    std::string source_meta_info = source_dir +"/meta.info.back";
    if (!fs::exists(dst_dir))
        fs::create_directories(dst_dir);
    
    if (!fs::exists(dst_so_file))
        fs::copy_file(source_so_file, dst_so_file);
    if (!fs::exists(dst_meta_file))
        fs::copy_file(source_meta_info, dst_meta_file);

    int node_id=1;
    JsonParam option;
    std::string module_type;
    std::string module_path;
    std::string module_entry;
    std::shared_ptr<Module> module;
    ModuleInfo module_info = ModuleFactory::create_module(module_name,node_id,option,module_type,module_path,module_entry,module);
    EXPECT_EQ(module == nullptr, 0);
    
}

TEST(module_factory, python_module) {
    std::string module_name = "test_python_module";
    std::string dst_dir = "/opt/tiger/bmf_mods/Module_"+module_name+"/";
    std::string dst_python_file = dst_dir + "my_module.py";
    std::string dst_meta_file = dst_dir + "meta.info";
    std::string source_dir = "../../../output/example/customize_module";
    std::string source_python_file = source_dir+"/my_module.py";
    std::string source_meta_info = source_dir +"/meta.info.back";
    if (!fs::exists(dst_dir))
        fs::create_directories(dst_dir);
    
    if (!fs::exists(dst_python_file))
        fs::copy_file(source_python_file, dst_python_file);
    if (!fs::exists(dst_meta_file))
        fs::copy_file(source_meta_info, dst_meta_file);

    //crete ___init__.py
    std::string init_py_path = dst_dir+"/__init__.py";
    std::ofstream fout(init_py_path);
    fout.close();
    int node_id=0;
    JsonParam option;
    std::string option_str = "{\"hello\":1}";
    option.parse(option_str);
    std::string module_type;
    std::string module_path;
    std::string module_entry;
    std::shared_ptr<Module> module;
    ModuleInfo module_info = ModuleFactory::create_module(module_name,node_id,option,module_type,module_path,module_entry,module);
    EXPECT_EQ(module == nullptr, 0);

}

#endif //BMF_ENABLE_MOBILE