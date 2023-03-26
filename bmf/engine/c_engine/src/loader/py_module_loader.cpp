#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <filesystem>
#include <bmf/sdk/module.h>
#include <bmf/sdk/log.h>
#include "../../../python/py_type_cast.h"


namespace py = pybind11;
namespace fs = std::filesystem;


namespace bmf_sdk{

/**
 * @brief Python Module Proxy 
 * 
 */
#pragma GCC visibility push(hidden) //remove visibility warning
class PyModule : public Module
{
    py::object self; // python module implementation 
public:
    template<typename...Args>
    py::object call_func(const char *name, Args&&...args)
    {
        if(!py::hasattr(self, name)){
            throw std::runtime_error(fmt::format("{} is not implemented", name));
        }

        return self.attr(name)(std::forward<Args>(args)...);
    }

    template<typename Ret, typename Func>
    Ret guard_call(const Func &f, Ret ok = 0, Ret nok = -1)
    {
        py::gil_scoped_acquire gil;
        try{
            f();
            return ok;
        }
        catch(std::exception &e){
            BMFLOG(BMF_WARNING) << e.what();
            return nok;
        }
    }

    PyModule(const py::object &cls, int32_t node_id = -1, JsonParam json_param = {})
    {
        py::gil_scoped_acquire gil;
        self = cls(node_id, json_param);
    }

    ~PyModule()
    {
        py::gil_scoped_acquire gil;
        self = py::object();
    }

    int32_t get_input_stream_info(JsonParam &json_param) override
    {
        return guard_call<int32_t>([&](){
            json_param = py::cast<JsonParam>(call_func("get_input_stream_info"));
        });
    }

    int32_t set_input_stream_info(JsonParam &json_param) override
    {
        return guard_call<int32_t>([&](){
            call_func("get_input_stream_info", json_param);
        });
    }


    int32_t get_output_stream_info(JsonParam &json_param) override
    {
        return guard_call<int32_t>([&](){
            json_param = py::cast<JsonParam>(call_func("get_output_stream_info"));
        });
    }

    int32_t set_output_stream_info(JsonParam &json_param) override
    {
        return guard_call<int32_t>([&](){
            call_func("get_output_stream_info", json_param);
        });
    }


    int32_t get_module_info(JsonParam &json_param) override
    {
        return guard_call<int32_t>([&](){
            json_param = py::cast<JsonParam>(call_func("get_module_info"));
        });
    }


    int32_t init() override
    {
        return guard_call<int32_t>([&](){
            call_func("init");
        });
    }

    int32_t reset() override
    {
        return guard_call<int32_t>([&](){
            call_func("reset");
        });
    }

    int32_t flush() override
    {
        return guard_call<int32_t>([&](){
            call_func("flush");
        });
    }


    int32_t dynamic_reset(JsonParam json_param) override
    {
        return guard_call<int32_t>([&](){
            call_func("dynamic_reset", json_param);
        });
    }


    int32_t process(Task &task) override
    {
        py::gil_scoped_acquire gil;
        auto task_obj = py::cast(task);
        auto ret = call_func("process", task_obj);
        task = py::cast<Task>(task_obj); //copy changes back
        if(ret.is_none()){
            throw std::runtime_error("PyModule.process return None");
        }
        else{
            return py::cast<int32_t>(ret); // enable exception throw
        }
    }

    int32_t close() override
    {
        return guard_call<int32_t>([&](){
            call_func("close");
        });
    }

    bool need_hungry_check(int input_stream_id) override
    {
        py::gil_scoped_acquire gil;
        return py::cast<bool>(
            call_func("need_hungry_check", input_stream_id));
    }

    bool is_hungry(int input_stream_id) override
    {
        py::gil_scoped_acquire gil;
        return py::cast<bool>(
            call_func("is_hungry", input_stream_id));
    }

    bool is_infinity() override
    {
        py::gil_scoped_acquire gil;
        return py::cast<bool>(
            call_func("is_infinity"));
    }

    //set_callback(...)
    void set_callback(std::function<CBytes(int64_t, CBytes)> callback_endpoint) override
    {
        py::gil_scoped_acquire gil;
        auto py_func = py::cpp_function([=](int64_t key, py::bytes &value){
            auto ret = callback_endpoint(key, py::cast<CBytes>(value));
            return py::cast(ret); //cbytes
        });
        call_func("set_callback", py_func);
    }


    bool is_subgraph() override
    {
        py::gil_scoped_acquire gil;
        return py::cast<bool>(
            call_func("is_subgraph"));
    }

    bool get_graph_config(JsonParam &json_param) override
    {
        return guard_call<int32_t>([&](){
            auto json_str = py::cast<std::string>(call_func("get_graph_config").attr("dump")());
            json_param = JsonParam(json_str);
        });
    }
};
#pragma GCC visibility pop


class PyModuleFactory : public ModuleFactoryI
{
public:
    using FactoryFunc = std::function<std::shared_ptr<Module>(int, const JsonParam&)>;

    PyModuleFactory(const FactoryFunc &factory) : factory_(factory)
    {
        sdk_version_ = BMF_SDK_VERSION;
    }

    std::shared_ptr<Module> make(int32_t node_id, const JsonParam &json_param) override
    {
        return factory_(node_id, json_param);
    }

    const std::string &sdk_version() const override
    {
        return sdk_version_;
    }

private:
    std::string sdk_version_;
    FactoryFunc factory_;
}; //


} //


//TODO: make it as a plugin
extern "C" bmf_sdk::ModuleFactoryI* bmf_import_py_module(
    const char *module_path, const char *module, const char *cls, char **errstr)
{
    if(!Py_IsInitialized()){
        //Py_Initialize(); //don't use it: ref:https://pybind11.readthedocs.io/en/stable/advanced/embedding.html
        py::initialize_interpreter();

        py::gil_scoped_release nogil;
        nogil.disarm(); //release gil
    }

    // try to use top most path as module path, which can fix import ambiguity
    // when module have same module.cls  
    std::string temp_module_path = fs::absolute(module_path);
    std::string temp_module_name = module;
    // if module_path last charector is '/' or '.', remove it
    while(temp_module_path.size() && (temp_module_path.back() == '/' || temp_module_path.back() == '.')){
        temp_module_path.pop_back();
    }
    while(fs::exists(fs::path(temp_module_path)/"__init__.py")){
        auto p = fs::path(temp_module_path);
        temp_module_name = std::string(p.filename()) + "." + temp_module_name;
        temp_module_path = p.parent_path();
    }

    //
    try{
        //check
        {
            py::gil_scoped_acquire gil;
            auto sys_path = py::module_::import("sys").attr("path").cast<py::list>(); 
            bool has = false;
            for(int i = 0; i < sys_path.size(); ++i){
                if(sys_path[i].cast<std::string>() == temp_module_path){
                    has = true;
                    break;
                }
            }

            if(!has){
                sys_path.append(temp_module_path);
            }
            py::module_::import(temp_module_name.c_str()).attr(cls);
        }

        //
        std::string cls_s(cls);
        auto module_factory = [=](int node_id, const bmf_sdk::JsonParam &json_param){
            py::gil_scoped_acquire gil;
            auto module_cls = py::module_::import(temp_module_name.c_str()).attr(cls_s.c_str());
            return std::make_shared<bmf_sdk::PyModule>(module_cls, node_id, json_param);
        };

        return new bmf_sdk::PyModuleFactory(module_factory);
    }
    catch(std::exception &e){
        if(errstr){
            *errstr = strdup(e.what());
        }
        BMFLOG(BMF_ERROR) << "Load python module " << module << " failed, " <<  e.what();
        return nullptr;
    }
}