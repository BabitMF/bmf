#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <hmp/tensor.h>
#include <bmf/sdk/json_param.h>
#include <bmf/sdk/cbytes.h>

namespace pybind11 { namespace detail {

template<> struct type_caster<bmf_sdk::JsonParam> {
    PYBIND11_TYPE_CASTER(bmf_sdk::JsonParam, _("JsonParam"));

    bool load(handle src, bool)
    {
        auto obj = src.ptr();

	    if(PyDict_Check(obj)){
            auto json = pybind11::module_::import("json");
            auto str = json.attr("dumps")(src).cast<std::string>();
            value = bmf_sdk::JsonParam(str);
            return true;
	    }
        else{
            throw std::runtime_error("Only support dict type");
        }
    }

    static handle cast(bmf_sdk::JsonParam src, return_value_policy, handle){
        std::string str = src.dump();
        pybind11::dict dict;
        if(str != "null"){
            auto json = pybind11::module_::import("json");
            dict = json.attr("loads")(pybind11::cast(str));
        }
        return dict.release(); //release!!
    }
};


template<> struct type_caster<bmf_sdk::CBytes> {
    PYBIND11_TYPE_CASTER(bmf_sdk::CBytes, _("CBytes"));

    bool load(handle src, bool)
    {
        auto obj = src.ptr();
        if(PyBytes_Check(obj)){
            //NOTE: Deep copy
            auto size = PyBytes_Size(obj);
            if(size){
                auto cb = bmf_sdk::CBytes::make(size);
                memcpy((void*)cb.buffer, PyBytes_AsString(obj), size);
                std::swap(cb, value);
            }
            else{
                value.buffer = nullptr;
                value.size = 0;
            }
            return true;
        }
        else{
            throw std::runtime_error("Only support bytes type");
        }
    }

    static handle cast(bmf_sdk::CBytes src, return_value_policy, handle){
        // NOTE: Deepcopy
        return pybind11::bytes((const char*)src.buffer, src.size).release();
    }
};


#define def_nogil(...) def(__VA_ARGS__, py::call_guard<py::gil_scoped_release>())


}}; //namespace pybind11::detail
