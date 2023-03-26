#include <fstream>

#include <bmf/sdk/log.h>
#include <bmf/sdk/exception_factory.h>
#include <bmf/sdk/error_define.h>
#include <bmf/sdk/module_registry.h>
#include <bmf/sdk/module_manager.h>
#include <bmf/sdk/shared_library.h>
#include <bmf/sdk/compat/path.h>

namespace bmf_sdk{



const char * s_bmf_sys_root = "/opt/tiger/bmf/";
const char * s_bmf_repo_root = "/opt/tiger/bmf_mods/";

static void string_split(std::vector<std::string> &tokens,
	const std::string &str, const std::string &seps)
{
    size_t j = 0;
    for(size_t i = 0; i < str.size(); ++i){
        if(seps.find(str[i]) != std::string::npos){
            if(i > j){
                tokens.push_back(str.substr(j, i - j));
            }
            j = i + 1;
        }
    }
    if(j < str.size()){
        tokens.push_back(str.substr(j));
    }
}


static std::string unique_name(const ModuleInfo &info)
{
    return fmt::format("{}:{}:{}", info.module_type, info.module_path, info.module_entry);
}

class CPPModuleFactory : public ModuleFactoryI
{
    SharedLibrary lib_;
    std::string class_name_;
    std::string sdk_version_;
public:
    CPPModuleFactory(const std::string &so, const std::string &class_name)
        : class_name_(class_name)
    {
        if(!so.empty()){ //in-app module have no .so file
            lib_ = SharedLibrary(so, SharedLibrary::LAZY | SharedLibrary::GLOBAL);
        }

        if (!ModuleRegistry::Registry().count(class_name)){
            auto msg = "Cannot find specified C++ module class: " + class_name;
            BMFLOG(BMF_ERROR) << msg << std::endl;
            throw std::logic_error(msg);
        }

        sdk_version_ = ModuleRegistry::GetModuleUsingSDKVersion(class_name_);
    }

    const std::string &sdk_version() const override
    {
        return sdk_version_;
    }

    std::shared_ptr<Module> make(int32_t node_id = -1,
                                         const JsonParam &json_param = {}) override
    {
        BMFLOG(BMF_INFO) << "Constructing c++ module" << std::endl;
        auto module = ModuleRegistry::ConstructModule(class_name_, node_id, json_param);
        BMFLOG(BMF_INFO) << "c++ module constructed" << std::endl;
        return module;
    }
};



struct ModuleManager::Private
{
    bmf_nlohmann::json builtin_config;
    std::string builtin_root;

    // cached module info
    std::map<std::string, ModuleInfo> known_modules;

    // cached moudle factories
    std::map<std::string, std::shared_ptr<ModuleFactoryI>> factories;

    // supported module loaders
    std::map<std::string, std::function<ModuleFactoryI*(const ModuleInfo&)>> loaders;
};


ModuleManager::ModuleManager()
{
    self = std::make_unique<Private>();

    //locate BUILTIN_CONFIG.json
    std::vector<std::string> roots;
    roots.push_back(s_bmf_sys_root);
    auto lib_path = fs::path(SharedLibrary::this_line_location()).parent_path();
    roots.push_back(lib_path);
    roots.push_back(lib_path.parent_path());
    roots.push_back(fs::current_path());

    auto fn = std::string("BUILTIN_CONFIG.json");
    for(auto &p : roots){
        auto fp = fs::path(p) / fn;
        if(fs::exists(fp) && !fs::is_directory(fp)){
            self->builtin_root = p;
            break;
        }
    }

    //
    fn = fs::path(self->builtin_root) / fn;
    if(!fs::exists(fn)){
        return;
    }
    std::ifstream fp(fn);
    fp >> self->builtin_config;

    // initialize cpp/py/go loader lazily
}


const ModuleInfo* ModuleManager::resolve_module_info(const std::string &module_name)
{
    // check if it has already cached
    if(self->known_modules.find(module_name) != self->known_modules.end()){
        return &self->known_modules.at(module_name);
    }

    // resolvers
    std::vector<decltype(&ModuleManager::resolve_from_builtin)> resolvers{
        &ModuleManager::resolve_from_builtin,
        &ModuleManager::resolve_from_meta
    };

    ModuleInfo info;
    for(auto resolver : resolvers){
        if((this->*resolver)(module_name, info)){
            self->known_modules[module_name] = info;
            return &self->known_modules.at(module_name);
        }
    }

    return nullptr;
}

std::shared_ptr<ModuleFactoryI> ModuleManager::load_module(const ModuleInfo &info, ModuleInfo *info_out)
{
    return load_module(info.module_name, info.module_type, 
                       info.module_path, info.module_entry, info_out);
}


std::shared_ptr<ModuleFactoryI> ModuleManager::load_module(
                            const std::string &module_name,
                            const std::string &module_type,
                            const std::string &module_path,
                            const std::string &module_entry,
                            ModuleInfo *info)
{
    // resolve module info
    auto tmp_module_info = resolve_module_info(module_name);
    ModuleInfo module_info;
    if(tmp_module_info == nullptr){
        //try load from local
        module_info.module_name = module_name;
        module_info.module_entry = module_name + "." + module_name;
        module_info.module_type = module_type.empty() ? infer_module_type(module_path) : module_type;
        if(module_info.module_type == "python"){
            module_info.module_path = fs::current_path();
        }
    }
    else{
        module_info = *tmp_module_info;
    }

    // merge module info
    if(!module_type.empty()){
        module_info.module_type = module_type;
    }
    if(!module_entry.empty()){
        module_info.module_entry = module_entry;
    }
    if(!module_path.empty()){
        module_info.module_path = module_path;
    }

    // check if it is cached
    auto module_id = unique_name(module_info);
    if(self->factories.find(module_id) != self->factories.end()){
        return self->factories.at(module_id);
    }

    //
    BMFLOG(BMF_INFO) << "Module info " << module_info.module_name << " " << module_info.module_type << " "
            << module_info.module_entry << " " << module_info.module_path << std::endl;

    if(!initialize_loader(module_info.module_type)){
        throw std::invalid_argument(
            fmt::format("Module type {} is not supported", module_info.module_type));
    }

    auto &loader = self->loaders.at(module_info.module_type);
    auto factory = std::shared_ptr<ModuleFactoryI>(loader(module_info));
    //python/go factory must deconstrut before main() return
    if(module_info.module_type == "c++"){
        self->factories[module_id] = factory;
    }

    if(info){
        *info = module_info;
    }
    return factory;
}


bool ModuleManager::resolve_from_builtin(const std::string &module_name, ModuleInfo &info) const
{
    if(!self->builtin_config.contains(module_name)){
        return false;
    }

    auto &vinfo = self->builtin_config[module_name];
    auto vget = [&](const std::string &key, const std::string &def){
        return vinfo.contains(key) ? vinfo[key].get<std::string>() : def;
    };
    auto module_path = vget("path", "");
    auto module_type = vget("type", "");
    auto module_class = vget("class", "");
    std::string module_file;
    if (module_type.empty())
    {
        throw std::invalid_argument("missing type in builtin config");
    }
    if (module_type != "python" && module_type != "c++" && module_type != "go")
    {
        throw std::invalid_argument("unsupported builtin module type.(must be c++/python/go");
    }

    if (module_path.empty())
    {
        if (module_type == "c++")
        {
            module_path = fs::path(self->builtin_root) / (std::string("lib/libbuiltin_modules") + SharedLibrary::default_extension());
            module_file = "libbuiltin_modules";
        }
        else if (module_type == "python")
        {
            module_path = fs::path(self->builtin_root) / std::string("python_builtins");
            if (!module_class.empty())
                module_file = module_class;
            else
                module_file = module_name;
        }
        else if (module_type == "go")
        {
            if (!module_class.empty())
            {
                module_path = fs::path(self->builtin_root) / std::string("lib") / (module_class + SharedLibrary::default_extension());
                module_file = module_class;
            }
            else
            {
                module_path = fs::path(self->builtin_root) / std::string("lib") / (module_name + SharedLibrary::default_extension());
                module_file = module_name;
            }
        }
    }
    if (module_class.empty())
        module_class = module_name;
    auto module_entry = module_file + "." + module_class;

    BMFLOG(BMF_INFO) << module_name << " " << module_type << " " << module_path << " " << module_entry
                     << std::endl;

    info = ModuleInfo(module_name, module_type, module_entry, module_path);
    return true;
}


bool ModuleManager::resolve_from_meta(const std::string &module_name, ModuleInfo &info) const
{
    std::vector<const char*> roots{
        s_bmf_repo_root,
        "./"
    };
    std::string meta_path;
    for(auto &r : roots){
        auto p = fs::path(r) / fmt::format("Module_{}", module_name) / std::string("meta.info");
        if(fs::exists(p)){
            meta_path = p;
            break;
        }
    }
    if(meta_path.empty()){
        return false;
    }

    //read meta file
    std::string pth, cls;
    JsonParam meta;
    meta.load(meta_path);

    std::string module_type = meta.get<std::string>("type");
    if (module_type == "PYTHON" || module_type == "python3"){
        info.module_type = "python";
    }
    if (module_type == "binary"){
        info.module_type = "c++";
    }
    if (module_type == "golang"){
        info.module_type = "go";
    }
    info.module_entry = meta.get<std::string>("entry");
    auto module_path = fs::path(meta_path).parent_path();

    //
    if (info.module_entry.empty()){
        info.module_entry = module_name + "." + module_name;
    }
    std::vector<std::string> entry_path;
    string_split(entry_path, info.module_entry, ".:");
    auto module_class = entry_path[entry_path.size() - 1];
    entry_path.pop_back();
    for(auto &e : entry_path){
        module_path /= e;
    }
    info.module_entry = entry_path[entry_path.size() - 1] + "." + module_class;

    //fix module path
    if (info.module_type == "python"){
        module_path = module_path.parent_path();
    }
    else if (info.module_type == "c++"){
        module_path =
            module_path.parent_path() /
            module_path.filename().replace_extension(SharedLibrary::default_extension());
    }
    else if (info.module_type == "go"){
        module_path =
            module_path.parent_path() /
            module_path.filename().replace_extension(SharedLibrary::default_extension());
    }
    info.module_path = module_path;
    info.module_name = module_name;

    return true;
}


bool ModuleManager::initialize_loader(const std::string &module_type)
{
    if(self->loaders.find(module_type) != self->loaders.end()){
        return true;
    }

    if(module_type == "c++"){
        self->loaders["c++"] = [&](const ModuleInfo &info) -> ModuleFactoryI*{
            std::string _, class_name;
            std::tie(_, class_name) = parse_entry(info.module_entry);
            return new CPPModuleFactory(info.module_path, class_name);
        };
        return true;
    }
    if(module_type == "python"){
        auto lib_name = std::string("libbmf_py_loader") + SharedLibrary::default_extension();
        auto loader_path = fs::path(SharedLibrary::this_line_location()).parent_path() / lib_name;
        auto lib = std::make_shared<SharedLibrary>(loader_path, 
                            SharedLibrary::LAZY | SharedLibrary::GLOBAL);

        self->loaders["python"] = [=](const ModuleInfo &info) -> ModuleFactoryI*{
            std::string module_file, class_name;
            std::tie(module_file, class_name) = parse_entry(info.module_entry);
            auto import_func = lib->symbol<ModuleFactoryI*(*)(const char*, const char*, const char*, char**)>(
                                                             "bmf_import_py_module");
            char *errstr = nullptr;
            auto mptr = import_func(info.module_path.c_str(),
                                         module_file.c_str(), class_name.c_str(), &errstr);
            if(errstr != nullptr){
                auto err = std::string(errstr);
                free(errstr);
                throw std::runtime_error(err);
            }
            return mptr;
        };
        return true;
    }
    else if(module_type == "go"){
        auto lib_name = std::string("libbmf_go_loader") + SharedLibrary::default_extension();
        auto loader_path = fs::path(SharedLibrary::this_line_location()).parent_path() / lib_name;
        auto lib = std::make_shared<SharedLibrary>(loader_path, 
                            SharedLibrary::LAZY | SharedLibrary::GLOBAL);

        self->loaders["go"] = [=](const ModuleInfo &info) -> ModuleFactoryI*{
            auto import_func = lib->symbol<ModuleFactoryI*(*)(const char*, const char*, char**)>("bmf_import_go_module");
            char *errstr = nullptr;
            auto mptr = import_func(info.module_path.c_str(), info.module_name.c_str(), &errstr);
            if(errstr != nullptr){
                auto err = std::string(errstr);
                free(errstr);
                throw std::runtime_error(err);
            }
            return mptr;
        };
        return true;
    }
    else{
        return false;
    }
}


std::tuple<std::string, std::string> ModuleManager::parse_entry(const std::string &module_entry)
{
    std::vector<std::string> entry_path;
    string_split(entry_path, module_entry, ".:");
    if (entry_path.size() < 2){
        BMF_Error_(BMF_StsBadArg, "module_entry: ", module_entry.c_str(), "is not satisfy");
    }
    auto module_file = entry_path[0];
    for (int i = 1; i < entry_path.size() - 1; i++)
    {
        module_file += "." + entry_path[i];
    }
    auto module_class = entry_path[entry_path.size() - 1];
    return std::make_tuple(module_file, module_class);
}


std::string ModuleManager::infer_module_type(const std::string &path)
{
    if (fs::path(path).extension() == SharedLibrary::default_extension()){
        if (SharedLibrary(path).raw_symbol("ConstructorRegister")){ //FIMXE:
            return "go";
        }
        else{
            return "c++";
        }
    }
    return "python";
}


ModuleManager &ModuleManager::instance()
{
    static ModuleManager m;
    return m; 
}



} //namespace bmf_sdk