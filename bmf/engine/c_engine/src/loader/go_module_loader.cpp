
#include <bmf/sdk/task.h>
#include <bmf/sdk/shared_library.h>
#include <bmf/sdk/module.h>
#include <bmf/sdk/log.h>

namespace bmf_sdk{
namespace {

/**
 * @brief Go Module Proxy 
 * 
 */
class GoModule : public Module
{
    int id_;
    std::shared_ptr<SharedLibrary> lib_;

    char*(*process_func_)(int32_t, Task*);
    char*(*init_func_)(int32_t);
    char*(*close_func_)(int32_t);
    char*(*reset_func_)(int32_t);
    char*(*module_info_func_)(int32_t);
    bool (*hungry_check_func_)(int32_t, int32_t);
    bool (*is_hungry_func_)(int32_t, int32_t);
    bool (*is_infinity_func_)(int32_t);

    template<typename F, typename ...Args>
    void check_result(F &f, Args&&...args)
    {
        auto cstr = f(std::forward<Args>(args)...);
        if(cstr != nullptr){
            auto str = std::string(cstr);
            free(cstr);
            throw std::runtime_error(str);
        }
    }

public:
    GoModule(int id, const std::shared_ptr<SharedLibrary> &lib)
        : id_(id), lib_(lib)
    {
        process_func_ = lib->symbol<decltype(process_func_)>("ModuleProcess");
        init_func_ = lib->symbol<decltype(init_func_)>("ModuleInit");
        reset_func_ = lib->symbol<decltype(reset_func_)>("ModuleReset");
        close_func_ = lib->symbol<decltype(reset_func_)>("ModuleClose");
        module_info_func_ = lib->symbol<decltype(module_info_func_)>("ModuleGetInfo");
        hungry_check_func_ = lib->symbol<decltype(hungry_check_func_)>("ModuleNeedHungryCheck");
        is_hungry_func_ = lib->symbol<decltype(is_hungry_func_)>("ModuleIsHungry");
        is_infinity_func_ = lib->symbol<decltype(is_infinity_func_)>("ModuleIsInfinity");
    }

    ~GoModule()
    {
        close();
    }

    int get_module_info(JsonParam &module_info) override
    {
        auto info = module_info_func_(id_);
        if(info != nullptr){
            auto sinfo = std::string(info);
            free(info);

            module_info.parse(sinfo);
        }

        return 0;
    }

    int process(Task &task) override
    {
        check_result(process_func_, id_, &task);
        return 0;
    }

    int32_t init() override
    {
        check_result(init_func_, id_);
        return 0;
    }

    int32_t reset() override
    {
        check_result(reset_func_, id_);
        return 0;
    }

    int32_t close() override
    {
        if(lib_){
            check_result(close_func_, id_);
        }
        return 0;
    }

    bool need_hungry_check(int input_stream_id) override
    {
        return hungry_check_func_(id_, input_stream_id);
    }

    bool is_hungry(int input_stream_id) override
    {
        return is_hungry_func_(id_, input_stream_id);
    }

    bool is_infinity() override
    {
        return is_infinity_func_(id_);
    }
};


class GoModuleFactory : public ModuleFactoryI
{
    std::shared_ptr<SharedLibrary> lib_;
    std::string cls_;
    std::string sdk_version_;
public:
    GoModuleFactory(const std::string &path, const std::string &cls)
    {
        lib_ = std::make_shared<SharedLibrary>(path);
        lib_->symbol<void(*)()>("ConstructorRegister")();
        cls_ = cls;

        auto cstr = lib_->symbol<const char*(*)()>("BmfSdkVersion")();
        sdk_version_ = std::string(cstr);
        free((void*)cstr);
    }

    const std::string &sdk_version() const override
    {
        return sdk_version_;
    }


    std::shared_ptr<Module> make(int32_t node_id, const JsonParam &json_param) override
    {
        auto option = json_param.dump();
        auto construct_func = lib_->symbol<int32_t(*)(const char*, int32_t, const char*)>("ModuleConstruct");
        auto id = construct_func(cls_.c_str(), node_id, option.c_str());
        if(id == -1){
            throw std::runtime_error("Consturct module " + cls_ + " failed");
        }
        else if(id == -2){
            throw std::runtime_error("Module " + cls_ + " not found");
        }
        else if(id < 0){
            throw std::runtime_error("Unknown error when construct module " + cls_);
        }

        return std::make_shared<GoModule>(id, lib_);
    }
}; //


}} // bmf_sdk


//TODO: make it as a plugin
extern "C" bmf_sdk::ModuleFactoryI* bmf_import_go_module(
    const char *module_path, const char *module, char **errstr)
{
    try{
        return new bmf_sdk::GoModuleFactory(module_path, module);
    }
    catch(std::exception &e){
        if(errstr){
            *errstr = strdup(e.what());
        }
        BMFLOG(BMF_ERROR) << "Load go module " << module << " failed, " <<  e.what();
        return nullptr;
    }
}