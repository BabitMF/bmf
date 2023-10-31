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
#include <bmf/sdk/config.h>

namespace py = pybind11;

void module_sdk_bind(py::module &m);
void engine_bind(py::module &m);

#ifdef BMF_ENABLE_FFMPEG
void bmf_ffmpeg_bind(py::module &m);
#endif

PYBIND11_MODULE(_bmf, m) {
    m.doc() = "Bytedance Media Framework";

    auto sdk = m.def_submodule("sdk");
    module_sdk_bind(sdk);

    auto engine = m.def_submodule("engine");
    engine_bind(engine);

    m.def("get_version", []() { return BMF_BUILD_VERSION; });

    m.def("get_commit", []() { return BMF_BUILD_COMMIT; });

#ifdef BMF_ENABLE_FFMPEG
    bmf_ffmpeg_bind(sdk);
#endif
}
