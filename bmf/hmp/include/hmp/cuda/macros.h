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
#pragma once

#include <hmp/core/logging.h>

namespace hmp {

#define HMP_CUDA_CHECK(exp)                                                    \
    do {                                                                       \
        auto __err = (exp);                                                    \
        if (__err != cudaSuccess) {                                            \
            cudaGetLastError();                                                \
            HMP_REQUIRE(__err == cudaSuccess, "CUDA error: {}",                \
                        cudaGetErrorString(__err));                            \
        }                                                                      \
    } while (0)

#define HMP_CUDA_CHECK_WRN(exp)                                                \
    do {                                                                       \
        auto __err = (exp);                                                    \
        if (__err != cudaSuccess) {                                            \
            cudaGetLastError();                                                \
            HMP_WRN("CUDA error: {}", cudaGetErrorString(__err));              \
        }                                                                      \
    } while (0)

namespace cuda {

const static int MaxDevices = 8;

} // namespace cuda

} // namespace hmp