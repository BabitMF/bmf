#include <map>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <bmf/sdk/config.h>

namespace py = pybind11;


void module_sdk_bind(py::module &m);
void engine_bind(py::module &m);

PYBIND11_MODULE(_bmf, m)
{
    m.doc() = "Bytedance Media Framework";

    auto sdk = m.def_submodule("sdk");
    module_sdk_bind(sdk);

    auto engine = m.def_submodule("engine");
    engine_bind(engine);

    m.def("get_version", [](){
        return BMF_BUILD_VERSION;
    });

    m.def("get_commit", [](){
        return BMF_BUILD_COMMIT;
    });

}