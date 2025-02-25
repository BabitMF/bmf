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
#include <fstream>

#include <bmf/sdk/log.h>
#include <bmf/sdk/exception_factory.h>
#include <bmf/sdk/error_define.h>
#include <bmf/sdk/module_registry.h>
#include <bmf/sdk/module_manager.h>
#include <bmf/sdk/shared_library.h>

namespace bmf_sdk {

#ifdef _WIN32
const fs::path s_bmf_repo_root = "C:\\Users\\Public\\bmf_mods";
#elif EMSCRIPTEN
const fs::path s_bmf_repo_root = "/bmf/";
#else
const fs::path s_bmf_repo_root = "/usr/local/share/bmf_mods/";
const fs::path s_bmf_mods_path = "/opt/tiger/bmf_mods/";
#endif

static void string_split(std::vector<std::string> &tokens,
                         const std::string &str, const std::string &seps) {
    size_t j = 0;
    for (size_t i = 0; i < str.size(); ++i) {
        if (seps.find(str[i]) != std::string::npos) {
            if (i > j) {
                tokens.push_back(str.substr(j, i - j));
            }
            j = i + 1;
        }
    }
    if (j < str.size()) {
        tokens.push_back(str.substr(j));
    }
}

static std::string unique_name(const ModuleInfo &info) {
    return fmt::format("{}:{}:{}:{}", info.module_type, info.module_path,
                       info.module_entry, info.module_revision);
}

class CPPModuleFactory : public ModuleFactoryI {
    SharedLibrary lib_;
    std::string class_name_;
    std::string sdk_version_;

  public:
    CPPModuleFactory(const std::string &so, const std::string &class_name)
        : class_name_(class_name) {
        if (!so.empty()) { // in-app module have no .so file
            lib_ =
                SharedLibrary(so, SharedLibrary::LAZY | SharedLibrary::GLOBAL);
        }

        if (!ModuleRegistry::Registry().count(class_name)) {
            auto msg = "Cannot find specified C++ module class: " + class_name;
            BMFLOG(BMF_ERROR) << msg << std::endl;
            throw std::logic_error(msg);
        }

        sdk_version_ = ModuleRegistry::GetModuleUsingSDKVersion(class_name_);
    }

    const std::string &sdk_version() const override { return sdk_version_; }

    const bool module_info(ModuleInfo &info) const override {
        std::string dump_func_symbol = "register_" + class_name_ + "_info";
        if (lib_.has(dump_func_symbol)) {
            auto dump_func =
                lib_.symbol<void (*)(ModuleInfo &)>(dump_func_symbol);
            dump_func(info);
            return true;
        }
        return false;
    }

    std::shared_ptr<Module> make(int32_t node_id = -1,
                                 const JsonParam &json_param = {}) override {
        BMFLOG(BMF_INFO) << "Constructing c++ module" << std::endl;
        auto module =
            ModuleRegistry::ConstructModule(class_name_, node_id, json_param);
        BMFLOG(BMF_INFO) << "c++ module constructed" << std::endl;
        return module;
    }
};

ModuleManager::ModuleManager() {
    if (false == inited) {
        init();
    }
}

bool ModuleManager::set_repo_root(const std::string &path) {
    std::lock_guard<std::mutex> guard(m_mutex);
    if (fs::exists(path)) {
        self->repo_roots.push_back(path);
    }
    return true;
}

std::function<ModuleFactoryI *(const ModuleInfo &)>
ModuleManager::get_loader(const std::string module_type) {
    if (self->loaders.find(module_type) == self->loaders.end()) {
        throw std::runtime_error("loader not found.");
    }
    return self->loaders.at(module_type);
}

const ModuleInfo *
ModuleManager::resolve_module_info(const std::string &module_name) {
    std::lock_guard<std::mutex> guard(m_mutex);

    // check if it has already cached
    if (self->known_modules.find(module_name) != self->known_modules.end()) {
        return &self->known_modules.at(module_name);
    }

    // resolvers
    std::vector<decltype(&ModuleManager::resolve_from_builtin)> resolvers{
        &ModuleManager::resolve_from_builtin,
        &ModuleManager::resolve_from_meta};

    ModuleInfo info;
    for (auto resolver : resolvers) {
        if ((this->*resolver)(module_name, info)) {
            self->known_modules[module_name] = info;
            return &self->known_modules.at(module_name);
        }
    }

    return nullptr;
}

const std::map<std::string, ModuleInfo> ModuleManager::resolve_all_modules() {
    std::lock_guard<std::mutex> guard(m_mutex);
    load_all_modules();
    return self->known_modules;
}

std::shared_ptr<ModuleFactoryI>
ModuleManager::load_module(const ModuleInfo &info, ModuleInfo *info_out) {
    return load_module(info.module_name, info.module_type, info.module_path,
                       info.module_entry, info_out);
}

std::shared_ptr<ModuleFactoryI>
ModuleManager::load_module(const std::string &module_name,
                           const std::string &module_type,
                           const std::string &module_path,
                           const std::string &module_entry, ModuleInfo *info) {
    // resolve module info
    auto tmp_module_info = resolve_module_info(module_name);
    ModuleInfo module_info;
    if (tmp_module_info == nullptr) {
        // try load from local
        module_info.module_name = module_name;
        module_info.module_entry = module_name + "." + module_name;
        module_info.module_type =
            module_type.empty() ? infer_module_type(module_path) : module_type;
        if (module_info.module_type == "python") {
            module_info.module_path = fs::current_path().string();
        }
    } else {
        module_info = *tmp_module_info;
    }

    std::lock_guard<std::mutex> guard(m_mutex);
    // merge module info
    if (!module_type.empty()) {
        module_info.module_type = module_type;
    }
    if (!module_entry.empty()) {
        module_info.module_entry = module_entry;
    }
    if (!module_path.empty()) {
        module_info.module_path = module_path;
    }

    // check if it is cached
    auto module_id = unique_name(module_info);
    if (self->factories.find(module_id) != self->factories.end()) {
        return self->factories.at(module_id);
    }

    BMFLOG(BMF_INFO) << "Module info " << module_info.module_name << " "
                     << module_info.module_type << " "
                     << module_info.module_entry << " "
                     << module_info.module_path << std::endl;

    if (!initialize_loader(module_info.module_type)) {
        throw std::invalid_argument(fmt::format(
            "Module type {} is not supported", module_info.module_type));
    }

    auto &loader = self->loaders.at(module_info.module_type);
    auto factory = std::shared_ptr<ModuleFactoryI>(loader(module_info));
    // python/go factory must deconstrut before main() return
    if (module_info.module_type == "c++") {
        self->factories[module_id] = factory;
    }

    if (info) {
        *info = module_info;
    }
    return factory;
}

void ModuleManager::load_all_modules() {
    std::vector<std::string> keys;
    for (const auto &item : self->builtin_config.items()) {
        keys.emplace_back(item.key());
    }

    ModuleInfo info;
    for (const auto &key : keys) {
        if (resolve_from_builtin(key, info)) {
            self->known_modules[info.module_name] = info;
        }
    }

    for (auto &root : self->repo_roots) {
        auto r = fs::path(root);
        std::string module_prefix = (r / "Module_").string();
        for (const auto &dir_entry : fs::directory_iterator(r)) {
            if (fs::is_directory(dir_entry) and
                dir_entry.path().string().rfind(module_prefix, 0) == 0) {
                std::string module_name =
                    dir_entry.path().string().erase(0, module_prefix.length());
                if (resolve_from_meta(module_name, info)) {
                    self->known_modules[info.module_name] = info;
                }
            }
        }
    }
}

bool ModuleManager::resolve_from_builtin(const std::string &module_name,
                                         ModuleInfo &info) const {
    if (!self->builtin_config.contains(module_name)) {
        return false;
    }

    auto &vinfo = self->builtin_config[module_name];
    auto vget = [&](const std::string &key, const std::string &def) {
        return vinfo.contains(key) ? vinfo[key].get<std::string>() : def;
    };
    auto module_path = vget("path", "");
    auto module_type = vget("type", "");
    auto module_class = vget("class", "");
    auto module_revision = vget("revision", "");
    std::string module_file;
    if (module_type.empty()) {
        throw std::invalid_argument("missing type in builtin config");
    }
    if (module_type != "python" && module_type != "c++" &&
        module_type != "go") {
        throw std::invalid_argument(
            "unsupported builtin module type.(must be c++/python/go");
    }

    if (module_path.empty()) {
        if (module_type == "c++") {
            module_path =
                (fs::path(self->builtin_root) /
                 std::string(SharedLibrary::default_shared_dir()) /
                 (std::string(SharedLibrary::default_prefix()) +
                  "builtin_modules" + SharedLibrary::default_extension()))
                    .string();
            module_file = std::string(SharedLibrary::default_prefix()) +
                          "builtin_modules";
        } else if (module_type == "python") {
            module_path =
                (fs::path(self->builtin_root) / std::string("python_builtins"))
                    .string();
            if (!module_class.empty())
                module_file = module_class;
            else
                module_file = module_name;
        } else if (module_type == "go") {
            if (!module_class.empty()) {
                module_path =
                    (fs::path(self->builtin_root) /
                     std::string(SharedLibrary::default_shared_dir()) /
                     (module_class + SharedLibrary::default_extension()))
                        .string();
                module_file = module_class;
            } else {
                module_path =
                    (fs::path(self->builtin_root) /
                     std::string(SharedLibrary::default_shared_dir()) /
                     (module_name + SharedLibrary::default_extension()))
                        .string();
                module_file = module_name;
            }
        }
    }
    if (module_class.empty())
        module_class = module_name;
    auto module_entry = module_file + "." + module_class;

    BMFLOG(BMF_INFO) << module_name << " " << module_type << " " << module_path
                     << " " << module_entry << std::endl;

    info = ModuleInfo(module_name, module_type, module_entry, module_path,
                      module_revision);
    return true;
}

bool ModuleManager::resolve_from_meta(const std::string &module_name,
                                      ModuleInfo &info) const {
    for (auto &r : self->repo_roots) {
        std::string meta_path;
        auto p = fs::path(r) / fmt::format("Module_{}", module_name) /
                 std::string("meta.info");
        if (fs::exists(p)) {
            meta_path = p.string();
        } else {
            continue;
        }

        // read meta file
        JsonParam meta;
        meta.load(meta_path);

        auto vget = [&](const std::string &key, const std::string &def) {
            return meta.has_key(key) ? meta.get<std::string>(key) : def;
        };

        // module_name
        if (info.module_name = vget("name", "");
            module_name != info.module_name) {
            BMFLOG(BMF_WARNING)
                << "The module_name in meta is: " << info.module_name
                << ", which is different from the actual module_name:"
                << module_name << std::endl;
        }

        // module_type
        auto meta_module_type = vget("type", "");
        if (meta_module_type == "python" || meta_module_type == "PYTHON" ||
            meta_module_type == "python3") {
            info.module_type = "python";
        } else if (meta_module_type == "binary" || meta_module_type == "c++") {
            info.module_type = "c++";
        } else if (meta_module_type == "golang" || meta_module_type == "go") {
            info.module_type = "go";
        } else {
            throw std::invalid_argument(
                "unsupported module type.(must be c++/python/go");
        }

        // module_class, module_entry
        auto meta_module_class = vget("class", "");
        info.module_entry = vget("entry", "");
        if (meta_module_class == "" && info.module_entry == "") {
            throw std::invalid_argument(
                "one of class or entry should be provided");
        }
        if (info.module_entry == "") {
            if (info.module_type == "c++") {
                info.module_entry =
                    std::string(SharedLibrary::default_prefix()) +
                    info.module_name + "." + meta_module_class;
            } else if (info.module_type == "python") {
                info.module_entry = meta_module_class + "." + meta_module_class;
            } else if (info.module_type == "go") {
                info.module_entry = info.module_name + "." + meta_module_class;
            }
            // BMFLOG(BMF_WARNING) << "Can not find entry from meta file, using
            // default: " << info.module_entry << std::endl;
        }
        std::vector<std::string> entry_path;
        string_split(entry_path, info.module_entry, ".:");
        if (entry_path.size() < 2) {
            BMF_Error_(BMF_StsBadArg,
                       "module_entry: ", info.module_entry.c_str(),
                       "is not satisfy");
        }
        if (auto entry_module_class = entry_path[entry_path.size() - 1];
            meta_module_class == "") {
            meta_module_class = entry_module_class;
        } else if (meta_module_class != entry_module_class) {
            BMFLOG(BMF_WARNING)
                << "The module class in meta is: " << meta_module_class
                << ", which is different from the entry field:"
                << entry_module_class << std::endl;
        }
        entry_path.pop_back();
        info.module_entry =
            entry_path[entry_path.size() - 1] + "." + meta_module_class;
        auto entry_module_path = fs::path(meta_path).parent_path();
        for (auto &e : entry_path) {
            entry_module_path /= e;
        }

        // module_path
        if (info.module_path = vget("path", "");
            info.module_path.empty()) { // builtin modules
            if (info.module_type == "c++" || info.module_type == "go") {
                info.module_path =
                    (entry_module_path.parent_path() /
                     entry_module_path.filename().replace_extension(
                         SharedLibrary::default_extension()))
                        .string();
            } else if (info.module_type == "python") {
                info.module_path = entry_module_path.parent_path().string();
            }
        }

        info.module_revision = vget("revision", "");

        // XXX: other attributes, such as envs...

        BMFLOG(BMF_INFO) << info.module_name << " " << info.module_type << " "
                         << info.module_path << " " << info.module_entry
                         << std::endl;
        return true;
    }
    return false;
}

bool ModuleManager::initialize_loader(const std::string &module_type) {
    if (self->loaders.find(module_type) != self->loaders.end()) {
        return true;
    }

    if (module_type == "c++") {
        self->loaders["c++"] = [&](const ModuleInfo &info) -> ModuleFactoryI * {
            std::string _, class_name;
            std::tie(_, class_name) = parse_entry(info.module_entry, false);
            return new CPPModuleFactory(info.module_path, class_name);
        };
        return true;
    }
    if (module_type == "python") {
        auto lib_name = std::string(SharedLibrary::default_prefix()) +
                        "bmf_py_loader" + SharedLibrary::default_extension();
        auto loader_path =
            fs::path(SharedLibrary::this_line_location())
                .lexically_normal().parent_path() / lib_name;
        auto lib = std::make_shared<SharedLibrary>(
            loader_path.string(), SharedLibrary::LAZY | SharedLibrary::GLOBAL);
        self->loaders["python"] =
            [=](const ModuleInfo &info) -> ModuleFactoryI * {
            std::string module_file, class_name;
            std::tie(module_file, class_name) =
                parse_entry(info.module_entry, false);
            auto import_func =
                lib->symbol<ModuleFactoryI *(*)(const char *, const char *,
                                                const char *, char **)>(
                    "bmf_import_py_module");
            char *errstr = nullptr;
            auto mptr =
                import_func(info.module_path.c_str(), module_file.c_str(),
                            class_name.c_str(), &errstr);
            if (errstr != nullptr) {
                auto err = std::string(errstr);
                free(errstr);
                throw std::runtime_error(err);
            }
            return mptr;
        };
        return true;
    } else if (module_type == "go") {
        auto lib_name = std::string(SharedLibrary::default_prefix()) +
                        "bmf_go_loader" + SharedLibrary::default_extension();
        auto loader_path =
            fs::path(SharedLibrary::this_line_location())
                .lexically_normal().parent_path() / lib_name;
        auto lib = std::make_shared<SharedLibrary>(
            loader_path.string(), SharedLibrary::LAZY | SharedLibrary::GLOBAL);

        self->loaders["go"] = [=](const ModuleInfo &info) -> ModuleFactoryI * {
            auto import_func =
                lib->symbol<ModuleFactoryI *(*)(const char *, const char *,
                                                char **)>(
                    "bmf_import_go_module");
            char *errstr = nullptr;
            auto mptr = import_func(info.module_path.c_str(),
                                    info.module_name.c_str(), &errstr);
            if (errstr != nullptr) {
                auto err = std::string(errstr);
                free(errstr);
                throw std::runtime_error(err);
            }
            return mptr;
        };
        return true;
    } else {
        return false;
    }
}

std::tuple<std::string, std::string>
ModuleManager::parse_entry(const std::string &module_entry, bool file_system) {
    std::vector<std::string> entry_path;
    string_split(entry_path, module_entry, ".:");
    if (entry_path.size() < 2) {
        BMF_Error_(BMF_StsBadArg, "module_entry: ", module_entry.c_str(),
                   "is not satisfy");
    }
    auto sep = file_system ? std::string{fs::path::preferred_separator} : ".";
    auto module_file = entry_path[0];
    for (int i = 1; i < entry_path.size() - 1; i++) {
        module_file += sep + entry_path[i];
    }
    auto module_class = entry_path[entry_path.size() - 1];
    return std::make_tuple(module_file, module_class);
}

std::string ModuleManager::infer_module_type(const std::string &path) {
    if (fs::path(path).extension() == SharedLibrary::default_extension()) {
        if (SharedLibrary(path).raw_symbol("ConstructorRegister")) { // FIXME:
            return "go";
        } else {
            return "c++";
        }
    }
    return "python";
}

void ModuleManager::init() {
    {
        std::lock_guard<std::mutex> guard(m_mutex);
        self = std::make_unique<Private>();

        // locate BUILTIN_CONFIG.json
        std::vector<std::string> roots;
        #ifndef EMSCRIPTEN
        auto lib_path = fs::path(SharedLibrary::this_line_location())
                            .lexically_normal().parent_path();
        roots.push_back(lib_path.string());
        roots.push_back(lib_path.parent_path().string());
        roots.push_back(fs::current_path().string());
        #else
        roots.push_back("/");
        #endif

        auto fn = std::string("BUILTIN_CONFIG.json");
        for (auto &p : roots) {
            auto fp = fs::path(p) / fn;
            if (fs::exists(fp) && !fs::is_directory(fp)) {
                self->builtin_root = p;
                break;
            }
        }

        fn = (fs::path(self->builtin_root) / fn).string();
        if (fs::exists(fn)) {
            std::ifstream fp(fn);
            fp >> self->builtin_config;
        } else {
            BMFLOG(BMF_ERROR) << "Module Mananger can not find:" << fn << "\n";
        }

        inited = true;
        // initialize cpp/py/go loader lazily
    }
    #ifndef EMSCRIPTEN
    set_repo_root((fs::path(SharedLibrary::this_line_location())
                       .lexically_normal()
                       .parent_path()
                       .parent_path() /
                   "cpp_modules")
                      .string());
    set_repo_root((fs::path(SharedLibrary::this_line_location())
                       .lexically_normal()
                       .parent_path()
                       .parent_path() /
                   "python_modules")
                      .string());
    set_repo_root((fs::path(SharedLibrary::this_line_location())
                       .lexically_normal()
                       .parent_path()
                       .parent_path() /
                   "go_modules")
                      .string());
    #endif
    set_repo_root(s_bmf_repo_root.string());
    set_repo_root(s_bmf_mods_path.string());
    set_repo_root(fs::current_path().string());
}

ModuleManager &ModuleManager::instance() {
    static ModuleManager m;
    return m;
}

} // namespace bmf_sdk
