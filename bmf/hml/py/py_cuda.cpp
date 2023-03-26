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
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <hmp/cuda/event.h>
#include <py_type_cast.h>

namespace py = pybind11;

using namespace hmp;


void cudaBind(py::module &m)
{
#ifdef HMP_ENABLE_CUDA
    auto cu = m.def_submodule("cuda");

    py::class_<cuda::Event>(cu, "Event")
        .def(py::init<bool, bool, bool>(), 
            py::arg("enable_timing"), py::arg("blocking")=true, py::arg("interprocess")=false)
        .def("is_created", &cuda::Event::is_created)
        .def("record", &cuda::Event::record, py::arg("stream")=py::none())
        .def("block", &cuda::Event::block, py::arg("stream")=py::none())
        .def("query", &cuda::Event::query)
        .def("synchronize", &cuda::Event::synchronize)
        .def("elapsed", &cuda::Event::elapsed)
    ;
#endif

}
