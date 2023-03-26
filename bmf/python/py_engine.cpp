#include <pybind11/pybind11.h>
#include <map>
#include "py_type_cast.h"
#include "connector.hpp"
#include "graph_config.h"
#include "optimizer.h"
#include "common.h"
#include <bmf/sdk/trace.h>

namespace py = pybind11;


void engine_bind(py::module &m)
{
    using namespace bmf;

    py::class_<BMFGraph>(m, "Graph")
        .def_nogil(py::init<std::string const&, bool, bool>(),
            py::arg("graph_config"), py::arg("is_path")=false, py::arg("need_merge")=true)
        .def_nogil("uid", &BMFGraph::uid)
        .def_nogil("start", &BMFGraph::start)
        .def_nogil("update", &BMFGraph::update, py::arg("config"), py::arg("is_path"))
        .def_nogil("close", &BMFGraph::close)
        .def_nogil("force_close", &BMFGraph::force_close)
        .def_nogil("add_input_stream_packet", &BMFGraph::add_input_stream_packet,
            py::arg("stream_name"), py::arg("packet"), py::arg("block")=false)
        .def_nogil("poll_output_stream_packet", &BMFGraph::poll_output_stream_packet,
            py::arg("stream_name"), py::arg("block")=true)
        .def_nogil("status", &BMFGraph::status)
    ;


    py::class_<BMFModule>(m, "Module")
        .def_nogil(py::init<std::string const&, std::string const &, std::string const&,
                      std::string const&, std::string const &>(),
            py::arg("module_name"), py::arg("option"), py::arg("module_type")="",
            py::arg("module_path")="", py::arg("module_entry")="")
        .def_nogil("uid", &BMFModule::uid)
        .def_nogil("process", &BMFModule::process, py::arg("task"))
        .def_nogil("reset", &BMFModule::reset)
        .def_nogil("init", &BMFModule::init)
        .def_nogil("close", &BMFModule::close)
    ;

    py::class_<BMFCallback>(m, "Callback")
        .def(py::init([](py::function &cb){
            return std::make_unique<BMFCallback>([=](bmf_sdk::CBytes para) -> bmf_sdk::CBytes{
                py::gil_scoped_acquire gil;
                auto res = cb(py::cast(para));
                return py::cast<bmf_sdk::CBytes>(res);
            });
        }))
        .def("uid", &BMFCallback::uid)
    ;

    // Trace interface
    py::enum_<bmf_sdk::TraceType>(m, "TraceType")
        .value("INTERLATENCY", bmf_sdk::TraceType::INTERLATENCY)
        .value("PROCESSING", bmf_sdk::TraceType::PROCESSING)
        .value("SCHEDULE", bmf_sdk::TraceType::SCHEDULE)
        .value("QUEUE_INFO", bmf_sdk::TraceType::QUEUE_INFO)
        .value("THROUGHPUT", bmf_sdk::TraceType::THROUGHPUT)
        .value("CUSTOM", bmf_sdk::TraceType::CUSTOM)
        .value("TRACE_START", bmf_sdk::TraceType::TRACE_START)
    ;

    py::enum_<bmf_sdk::TracePhase>(m, "TracePhase")
        .value("NONE", bmf_sdk::TracePhase::NONE)
        .value("START", bmf_sdk::TracePhase::START)
        .value("END", bmf_sdk::TracePhase::END)
    ;

    m.def_nogil("trace", (void(*)(bmf_sdk::TraceType, const char*, bmf_sdk::TracePhase, const char*))&bmf_sdk::BMF_TRACE,
         py::arg("category"), py::arg("name"), 
         py::arg("phase")=bmf_sdk::TracePhase::NONE,
         py::arg("str")=__builtin_FUNCTION());

    m.def_nogil("trace_info", (void(*)(bmf_sdk::TraceType, const char*, bmf_sdk::TracePhase, std::string, const char*))&bmf_sdk::BMF_TRACE,
         py::arg("category"), py::arg("name"),
         py::arg("phase")=bmf_sdk::TracePhase::NONE,
         py::arg("info")=std::string(),
         py::arg("str")=__builtin_FUNCTION());

    m.def_nogil("trace_done", [](){
        bmf_sdk::TraceLogger::instance()->format_logs(); 
    });

    m.def_nogil("convert_filter_para", [](const std::string &config){
        JsonParam json_config = JsonParam(config);
        bmf_engine::NodeConfig node_config = bmf_engine::NodeConfig(json_config);
        bmf_engine::Optimizer::convert_filter_para(node_config);
        bmf_engine::Optimizer::replace_stream_name_with_id(node_config);
        return node_config.get_option().dump();
    });

} //