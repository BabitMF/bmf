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
#include <memory>
#include <string>
#include <unordered_map>
#include <iostream>

#if defined(__GNUC__) && !defined(__clang__)
#include <features.h>

#if __GNUC_PREREQ(8, 1)
#include <filesystem>
#else

#include <experimental/filesystem>

namespace std {
namespace filesystem = experimental::filesystem;
}
#endif
#else

#include <filesystem>

#endif

#include <bmf/sdk/common.h>
#include <bmf/sdk/module.h>
#include <bmf/sdk/module_manager.h>
#include <bmf/sdk/json_param.h>

#ifndef BMF_ENGINE_MODULE_REGISTRY_H
#define BMF_ENGINE_MODULE_REGISTRY_H

BEGIN_BMF_SDK_NS

class BMF_API ModuleRegistry {
  public:
    typedef std::shared_ptr<Module> (*Constructor)(int node_id,
                                                   JsonParam json_param);

    typedef std::unordered_map<std::string, std::pair<std::string, Constructor>>
        ConstructorRegistry;

    static ConstructorRegistry &Registry();

    static void AddConstructor(std::string const &module_name,
                               std::string const &sdk_version,
                               Constructor constructor);

    static std::shared_ptr<Module>
    ConstructModule(std::string const &module_name, int node_id = -1,
                    JsonParam json_param = JsonParam());

    static std::string GetModuleUsingSDKVersion(std::string const &module_name);

  private:
    ModuleRegistry() {}
};

class BMF_API ModuleRegister {
  public:
    ModuleRegister(std::string const &module_name,
                   std::string const &sdk_version,
                   std::shared_ptr<Module> (*constructor)(
                       int node_id, JsonParam json_param));

    // For modules using old sdk.
    ModuleRegister(std::string const &module_name,
                   std::shared_ptr<Module> (*constructor)(
                       int node_id, JsonParam json_param));
};

#define REGISTER_MODULE_CONSTRUCTOR(module_name, constructor)                  \
    static ::bmf_sdk::ModuleRegister r_constructor_##module_name(              \
        #module_name, BMF_SDK_VERSION, constructor);
// HMP_DEFINE_TAG(r_constructor_##module_name);

#define REGISTER_MODULE_CLASS(module_name)                                     \
    std::shared_ptr<::bmf_sdk::Module> Constructor_##module_name##Module(      \
        int node_id, ::bmf_sdk::JsonParam json_param) {                        \
        return std::shared_ptr<::bmf_sdk::Module>(                             \
            new module_name(node_id, json_param));                             \
    }                                                                          \
    REGISTER_MODULE_CONSTRUCTOR(module_name, Constructor_##module_name##Module);

#define REGISTER_MODULE_INFO(module_name, info)                                \
    extern "C" BMF_API void register_##module_name##_info(ModuleInfo &info)

// make sure static module library is linked
#define BMF_DECLARE_MODULE(module_name)                                        \
    HMP_DECLARE_TAG(r_constructor_##module_name);
#define BMF_IMPORT_MODULE(module_name)                                         \
    HMP_IMPORT_TAG(r_constructor_##module_name);

END_BMF_SDK_NS

#endif // BMF_ENGINE_MODULE_REGISTRY_H
