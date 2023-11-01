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
#pragma once

#include <bmf/sdk/module.h>
#include <bmf/sdk/module_tag.h>
#include <bmf/sdk/compat/path.h>
#include <mutex>

namespace bmf_sdk {

extern const fs::path BMF_API s_bmf_repo_root;

class BMF_API ModuleInfo {
  public:
    ModuleInfo(){};

    // module_name: used to identify a module, is unique and used in graph
    // configuration
    // module_revision: the version of a module
    // module_type: the runtime type of a module, right now it could be
    // python/c++/go, by default it's 'python'
    // module_path: the full path to module file, should be directory for
    // python, or file path for c++/go
    // module_entry: the entry contains two parts: file_name and class_name,
    // they are connected using a dot;
    //               the file_name in entry doesn't contain the type extension
    //               name, it may looks duplicate with the last part of
    //               module_path
    ModuleInfo(const std::string &name, const std::string &type,
               const std::string &entry, const std::string &path = {},
               const std::string &revision = "",
               const std::string &description = "",
               const ModuleTag &tag = ModuleTag::BMF_TAG_NONE)
        : module_name(name), module_type(type), module_entry(entry),
          module_path(path), module_revision(revision),
          module_description(description), module_tag(tag){};

    ModuleInfo(const std::string &description, const ModuleTag &tag)
        : module_description(description), module_tag(tag){};

    explicit ModuleInfo(const std::string &name, const std::string &path = {})
        : ModuleInfo(name, {}, {}, path){};

    std::string module_name;
    std::string module_revision;
    std::string module_type;
    std::string module_entry;
    std::string module_path;
    std::string module_description;
    ModuleTag module_tag;
};

/**
 * @brief Module factory interface class
 *
 */
class ModuleFactoryI {
  public:
    virtual const std::string &sdk_version() const = 0;

    virtual std::shared_ptr<Module> make(int32_t node_id = -1,
                                         const JsonParam &json_param = {}) = 0;

    virtual const bool module_info(ModuleInfo &info) const = 0;

    virtual ~ModuleFactoryI() {}
}; //

//
class BMF_API ModuleManager {
    struct Private {
        nlohmann::json builtin_config;
        std::string builtin_root;
        std::vector<std::string> repo_roots;
        // cached module info
        std::map<std::string, ModuleInfo> known_modules;

        // cached moudle factories
        std::map<std::string, std::shared_ptr<ModuleFactoryI>> factories;

        // supported module loaders
        std::map<std::string,
                 std::function<ModuleFactoryI *(const ModuleInfo &)>>
            loaders;
    };

  public:
    // find module info from builtin and sys repo(/opt/tiger/xxx)
    const ModuleInfo *resolve_module_info(const std::string &module_name);

    const std::map<std::string, ModuleInfo> resolve_all_modules();

    // info -> in/out
    std::shared_ptr<ModuleFactoryI> load_module(const ModuleInfo &info,
                                                ModuleInfo *info_out = nullptr);

    std::shared_ptr<ModuleFactoryI> load_module(
        const std::string &module_name, const std::string &module_type = {},
        const std::string &module_path = {},
        const std::string &module_entry = {}, ModuleInfo *info = nullptr);

    bool set_repo_root(const std::string &path);

    std::function<ModuleFactoryI *(const ModuleInfo &)>
    get_loader(const std::string module_type);

    std::tuple<std::string, std::string>
    parse_entry(const std::string &module_entry, bool file_system);

    static ModuleManager &instance();

    // for the ModuleFunctor of jni
    std::mutex m_mutex_jni;
    // for the load_module function of ModuleManager
    std::mutex m_mutex;
    bool inited = false;

  protected:
    ModuleManager();
    void init();

    void load_all_modules();
    bool resolve_from_builtin(const std::string &module_name,
                              ModuleInfo &info) const;
    bool resolve_from_meta(const std::string &module_name,
                           ModuleInfo &info) const;
    bool initialize_loader(const std::string &module_type);

    std::string infer_module_type(const std::string &path);

  private:
    std::unique_ptr<Private> self;
};

} // namespace bmf_sdk
