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

#include <hmp/core/allocator.h>
#include <hmp/cuda/allocator.h>
#include <hmp/core/device.h>
#include <hmp/core/stream.h>
#include <hmp/tensor.h>
#include <map>
#include <thread>
#include <gtest/gtest.h>
#include <hmp/core/ref_ptr.h>


namespace hmp{

#ifdef HMP_ENABLE_CUDA
static bool has_cuda()
{
    return device_count(kCUDA) > 0;
}


TEST(TestAllocator, cuda_device_allocator)
{
    if(!has_cuda()){
#ifdef GTEST_SKIP
        GTEST_SKIP() << "No cuda device";
#else
        return;
#endif
    }

    srand(42);
    auto ndev = device_count(kCUDA);

    std::vector<std::pair<DataPtr, int64_t>> datas;
    std::vector<int64_t> sizes(ndev, 0); //for each device
    std::vector<int64_t> segs(ndev, 0); //for each device

    const int max_size = 4<<20;
    const int N = ndev * 1024;

    //
    for(int i = 0; i < ndev; ++i){
        auto stats = cuda::device_memory_stats(i);
        ASSERT_EQ(0, stats.active.current);
        ASSERT_EQ(0, stats.segment.current);
    }

    //random alloc and free
    for(int i = 0; i < N; ++i){
        auto did = rand() % ndev;
        int64_t size = (rand()*int64_t(rand())) % max_size;
        size = size == 0 ? 1 : size;
        bool replace = rand()%2;

        DeviceGuard d(Device(kCUDA, did));
        auto ptr = get_allocator(kCUDA)->alloc(size);

        if(replace && datas.size()){
            auto ridx = rand() % datas.size();
            auto &rptr = datas[ridx];
            sizes[rptr.first.device().index()] -= rptr.second;
            segs[rptr.first.device().index()] -= 1;
            datas[ridx] = std::make_pair(std::move(ptr), size);
        }
        else{
            datas.push_back(std::make_pair(std::move(ptr), size));
        }

        sizes[did] += size;
        segs[did] += 1;
    }

    //check
    std::this_thread::sleep_for(std::chrono::milliseconds(10)); //wait cudaEvent
    for(int i = 0; i < ndev; ++i){
        auto stats = cuda::device_memory_stats(i);
#if 0
        //actually, we can't track allocated bytes
        // as allocator do rounding and split
        EXPECT_EQ(sizes[i], stats.active.current);
#endif
        EXPECT_EQ(segs[i], stats.segment.current);
    }

    //
    datas.clear();
    std::this_thread::sleep_for(std::chrono::milliseconds(10)); //wait cudaEvent
    for(int i = 0; i < ndev; ++i){
        auto stats = cuda::device_memory_stats(i);
        EXPECT_EQ(0, stats.active.current);
        ASSERT_EQ(0, stats.segment.current);
    }
}


TEST(TestAllocator, cuda_device_allocator_inactive)
{
    if(!has_cuda()){
#ifdef GTEST_SKIP
        GTEST_SKIP() << "No cuda device";
#else
        return;
#endif
    }

    auto stream = create_stream(kCUDA);

    auto a = empty({8<<20}, TensorOptions(kCPU).pinned_memory(true));

    {
        StreamGuard g(stream);
        auto b = empty_like(a, TensorOptions(kCUDA));
        auto c = empty_like(b, TensorOptions(kCPU).pinned_memory(true));

        copy(b, a);
        copy(c, b);
    }
    auto cpu_stats0 = cuda::host_memory_stats();
    auto cuda_stats0 = cuda::device_memory_stats(0);

    EXPECT_FALSE(stream.query()); //ensure copy is not finished

    stream.synchronize();

    auto cpu_stats1 = cuda::host_memory_stats();
    auto cuda_stats1 = cuda::device_memory_stats(0);

    //
    EXPECT_EQ(a.nbytes(), cpu_stats0.inactive.current);
    EXPECT_EQ(a.nbytes(), cuda_stats0.inactive.current);

    EXPECT_EQ(0, cpu_stats1.inactive.current); //
    EXPECT_EQ(0, cuda_stats1.inactive.current); //
}

#endif


} //namespace