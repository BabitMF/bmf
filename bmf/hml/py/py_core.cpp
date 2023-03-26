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

#include <pybind11/pybind11.h>
#include <hmp/core/logging.h>
#include <hmp/core/scalar_type.h>
#include <hmp/core/device.h>
#include <hmp/core/stream.h>
#include <hmp/core/timer.h>
#include <py_type_cast.h>


namespace py = pybind11;

using namespace hmp;

void coreBind(py::module &m)
{
    py::enum_<ScalarType>(m, "ScalarType")
    #define DEF_TYPE_ATTR(_, name) \
        .value("k"#name, ::ScalarType::name)
        HMP_FORALL_SCALAR_TYPES(DEF_TYPE_ATTR)
    #undef DEF_TYPE_ATTR
        //Numpy like dtype names
        .value("uint8", kUInt8)
        .value("int8", kInt8)
        .value("uint16", kUInt16)
        .value("int16", kInt16)
        .value("int32", kInt32)
        .value("int64", kInt64)
        .value("float32", kFloat32)
        .value("float64", kFloat64)
        .export_values();

    py::enum_<DeviceType>(m, "DeviceType")
        .value("kCPU", kCPU)
        .value("kCUDA", kCUDA)
        .export_values();

    py::class_<Device>(m, "Device", py::dynamic_attr())
        .def(py::init<>())
        .def(py::init<DeviceType, Device::Index>(), py::arg("device_type"), py::arg("index") = 0)
        .def(py::init<std::string>())
        .def("__eq__", &Device::operator==)
        .def("__neq__", &Device::operator!=)
        .def("__repr__", [](const Device &device) { return stringfy(device); })
        .def("type", &Device::type)
        .def("index", &Device::index)
        .def("__enter__", [](py::object &self){
            auto device = self.cast<Device>();
            auto guard = DeviceGuard(device);
            self.attr("__guard__")  = py::cast(std::move(guard));
            return self;
        })
        .def("__exit__", [](py::object &self, py::args){
            self.attr("__guard__")  = py::none();
        })
    ;
    py::class_<DeviceGuard>(m, "DeviceGuard");

    py::class_<Stream>(m, "Stream", py::dynamic_attr())
        .def("__eq__", &Stream::operator==)
        .def("__neq__", &Stream::operator!=)
        .def("__repr__", [](const Stream &stream){
            return stringfy(stream);
        })
        .def("query", &Stream::query)
        .def("synchronize", &Stream::synchronize)
        .def("device", &Stream::device)
        .def("handle", &Stream::handle)
        .def("__enter__", [](py::object &self){
            auto stream = self.cast<Stream>();
            auto guard = StreamGuard(stream);
            self.attr("__guard__") = py::cast(std::move(guard));
            return self;
        })
        .def("__exit__", [](py::object &self, py::args){
            self.attr("__guard__") = py::none();
        })
    ;

    py::class_<StreamGuard>(m, "StreamGuard");

    m.def("device_count", &device_count, py::arg("device_type"));
    m.def("current_device", &current_device, py::arg("device_type"));
    m.def("set_current_device", &set_current_device, py::arg("device"));

    m.def("create_stream", &create_stream, py::arg("device_type"), py::arg("flags") = 0);
    m.def("current_stream", &current_stream, py::arg("device_type"));

    py::class_<Timer>(m, "Timer")
        .def("__repr__", (std::string(*)(const Timer&))&stringfy)
        .def("start", &Timer::start)
        .def("stop", &Timer::stop)
        .def("elapsed", &Timer::elapsed)
        .def("is_stopped", &Timer::is_stopped)
        .def("device", &Timer::device)
    ;

    m.def("create_timer", &create_timer, py::arg("device_type")=kCPU);
}