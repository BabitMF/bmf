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

#include <hmp/core/macros.h>

namespace hmp {
namespace cuda {

struct MemoryStat {
    int64_t current = 0;
    int64_t peak = 0;
    int64_t allocated = 0;
    int64_t freed = 0;
};

struct DeviceMemoryStats {
    //
    MemoryStat active;

    // unallocated but can't released via cudaFree
    MemoryStat inactive;

    // count blocks allocated by cudaMalloc or cudaMallocHost
    MemoryStat segment;
};

HMP_API DeviceMemoryStats device_memory_stats(int device);
HMP_API DeviceMemoryStats host_memory_stats();

HMP_API int d2d_memcpy(void *dst, size_t dpitch, const void *src, size_t spitch,
                       size_t width, size_t height);
} // namespace cuda
} // namespace hmp
