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

#include <map>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <hmp/core/macros.h>

namespace py = pybind11;

void coreBind(py::module &m);
void tensorBind(py::module &m);
void ffmpegBind(py::module &m);
void imageBind(py::module &m);

void cudaBind(py::module &m);

static std::map<const char*, int> sHMPConfigs{
#define DEF_CONFIG(M, value) std::make_pair(#M, value),
#ifdef HMP_ENABLE_CUDA
    DEF_CONFIG(HMP_ENABLE_CUDA, 1)
#else
    DEF_CONFIG(HMP_ENABLE_CUDA, 0)
#endif

#ifdef HMP_ENABLE_FFMPEG
    DEF_CONFIG(HMP_ENABLE_FFMPEG, 1)
#else
    DEF_CONFIG(HMP_ENABLE_FFMPEG, 0)
#endif

#ifdef HMP_ENABLE_OPENCV
    DEF_CONFIG(HMP_ENABLE_OPENCV, 1)
#else
    DEF_CONFIG(HMP_ENABLE_OPENCV, 0)
#endif

#ifdef HMP_ENABLE_NPP
    DEF_CONFIG(HMP_ENABLE_NPP, 1)
#else
    DEF_CONFIG(HMP_ENABLE_NPP, 0)
#endif

#ifdef HMP_ENABLE_OPENMP
    DEF_CONFIG(HMP_ENABLE_OPENMP, 1)
#else
    DEF_CONFIG(HMP_ENABLE_OPENMP, 0)
#endif

#ifdef HMP_ENABLE_TORCH
    DEF_CONFIG(HMP_ENABLE_TORCH, 1)
#else
    DEF_CONFIG(HMP_ENABLE_TORCH, 0)
#endif

#undef DEF_CONFIG
};



PYBIND11_MODULE(_hmp, m)
{
    m.doc() = "Python binding for hmp library";

    m.attr("__version__") = HMP_VERSION_STR();
    
    m.attr("__config__") = sHMPConfigs;

    //core modules
    coreBind(m);
    tensorBind(m);

    //sub modules
    imageBind(m);

    ffmpegBind(m);

    cudaBind(m);
}