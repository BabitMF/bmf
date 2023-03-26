#pragma once

#include <bmf/sdk/module.h>

namespace bmf_sdk{


class BMF_API ModuleInfo {
public:
    ModuleInfo() {};

    // module_name: used to identify a module, is unique and used in graph configuration
    // module_type: the runtime type of a module, right now it could be python/c++/go, by default it's 'python'
    // module_path: the full path to module file, should be directory for python, or file path for c++/go
    // module_entry: the entry contains two parts: file_name and class_name, they are connected using a dot;
    //               the file_name in entry doesn't contain the type extension name, it may looks duplicate with the last part of module_path
    ModuleInfo(const std::string &name, 
               const std::string &type,
               const std::string &entry,
               const std::string &path = {})
        : module_name(name), module_type(type),
          module_entry(entry), module_path(path)
    {
    };

    explicit ModuleInfo(const std::string &name, 
               const std::string &path = {}) 
        : ModuleInfo(name, {}, {}, path)
    {
    };

    std::string module_name;
    std::string module_type;
    std::string module_entry;
    std::string module_path;
};

//
class BMF_API ModuleManager
{
    struct Private;

public:
    // find module info from builtin and sys repo(/opt/tiger/xxx)
    const ModuleInfo* resolve_module_info(const std::string &module_name);
    
    // info -> in/out
    std::shared_ptr<ModuleFactoryI> load_module(
        const ModuleInfo &info, ModuleInfo *info_out = nullptr);

    std::shared_ptr<ModuleFactoryI> load_module(
                            const std::string &module_name,
                            const std::string &module_type = {},
                            const std::string &module_path = {},
                            const std::string &module_entry = {},
                            ModuleInfo *info = nullptr);

    static ModuleManager &instance();

protected:
    ModuleManager();

    bool resolve_from_builtin(const std::string &module_name, ModuleInfo &info) const;
    bool resolve_from_meta(const std::string &module_name, ModuleInfo &info) const;
    bool initialize_loader(const std::string &module_type);

    std::string infer_module_type(const std::string &path);

    std::tuple<std::string, std::string> parse_entry(const std::string &module_entry);
private:
    std::unique_ptr<Private> self;
};


} //namespace