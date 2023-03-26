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

#include <hmp/tensor.h>
#include <benchmark/benchmark.h>

using namespace hmp;

namespace {


template<DeviceType device, bool pinned>
void BM_allocator(benchmark::State &state)
{
    auto seed = state.range(0);
    auto max_size = state.range(1);
    auto options = TensorOptions(kFloat32).device(device).pinned_memory(pinned);

    std::vector<Tensor> used;
    for(auto _ : state){
        auto size = (rand()*long(rand())) % max_size;
        size = std::max<long>(size, 1);

        auto data = empty({size}, options);
        if((size&0x1) ==0x0){ //keep half of the data
            used.push_back(data);
        }
    }
}


BENCHMARK_TEMPLATE(BM_allocator, kCPU, false)
    ->Args({42, 4<<20})->Threads(8)->Unit(benchmark::kMicrosecond);

#ifdef HMP_ENABLE_CUDA
BENCHMARK_TEMPLATE(BM_allocator, kCPU, true)
    ->Args({42, 4<<20})->Threads(8)->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_allocator, kCUDA, false)
    ->Args({42, 4<<20})->Threads(8)->Unit(benchmark::kMicrosecond);
#endif



} //namespace