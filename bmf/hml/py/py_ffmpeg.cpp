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

#ifdef HMP_ENABLE_FFMPEG
#include <hmp/ffmpeg/ffmpeg.h>
#endif

#include <py_type_cast.h>

namespace py = pybind11;



void ffmpegBind(py::module &m)
{
#ifdef HMP_ENABLE_FFMPEG
    using namespace hmp;

    auto ff = m.def_submodule("ffmpeg");

    py::class_<ffmpeg::VideoReader>(ff, "VideoReader")
        .def(py::init<std::string>())
        .def("read", &ffmpeg::VideoReader::read);

    py::class_<ffmpeg::VideoWriter>(ff, "VideoWriter")
        .def(py::init<std::string, int, int, int, const PixelInfo&, int>(), 
            py::arg("fn"), py::arg("width"), py::arg("height"), py::arg("fps"), py::arg("pix_info"), py::arg("kbs")=2000)
        .def("write", &ffmpeg::VideoWriter::write);
#endif
}
