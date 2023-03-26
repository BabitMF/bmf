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

#include <cuda_runtime.h>
#include <hmp/core/stream.h>
#include <hmp/cuda/macros.h>
#include <hmp/cuda/allocator.h>
#include <hmp/core/allocator.h>
#include <unordered_set>
#include <mutex>
#include <set>
#include <thread>
#include <deque>

namespace hmp{
namespace cuda{

namespace {


constexpr size_t kMinBlockSize = 512;       // all sizes are rounded to at least 512 bytes
constexpr size_t kSmallSize = 1048576;      // largest "small" allocation is 1 MiB
constexpr size_t kSmallBuffer = 2097152;    // "small" allocations are packed in 2 MiB blocks
constexpr size_t kLargeBuffer = 20971520;   // "large" allocations may be packed in 20 MiB blocks
constexpr size_t kMinLargeAlloc = 10485760; // allocations between 1 and 10 MiB may use kLargeBuffer
constexpr size_t kRoundLarge = 2097152;     // round up large allocs to 2 MiB


void update_stat(MemoryStat& stat, int64_t size)
{

    stat.current += size;
    HMP_REQUIRE(stat.current >= 0, "Negtive amount of memory detected {} {}", stat.current, size);

    stat.peak = std::max(stat.current, stat.peak);
    if(size > 0){
        stat.allocated += size;
    }
    else{
        stat.freed -= size;
    }
}


using MallocFunc = cudaError_t(*)(void**, size_t);
using FreeFunc = cudaError_t(*)(void*);

struct Block;
using BlockComparison = bool(*)(const Block*, const Block*);
using BlockPool = std::set<Block*, BlockComparison>;


struct Block{
    Block(int64_t size_)
        : size(size_)
    {
    }

    Block(int device_, int64_t size_, void* ptr_, BlockPool *pool_)
        : device(device_), size(size_), ptr(ptr_), pool(pool_)
    {
    }

    bool is_split() const
    {
        return (prev != nullptr) || (next != nullptr);
    }

    int device = 0;
    int event_count = 0;
    int64_t size;
    void *ptr = nullptr;
    BlockPool *pool = nullptr;
    Block *prev = nullptr;
    Block *next = nullptr;
    bool allocated = false;

    std::set<cudaStream_t> streams;
};


static bool block_comparator(const Block *a, const Block *b)
{
    if(a->size != b->size){
        return a->size < b->size;
    }
    return a->ptr < b->ptr;
}

class CUDAAllocator : public Allocator
{
    DeviceMemoryStats stats_;

    mutable std::recursive_mutex mutex_;
    BlockPool small_blocks_;
    BlockPool large_blocks_;

    std::unordered_map<void*, Block*> alloced_;
    std::deque<std::pair<cudaEvent_t, Block*>> freed_;

    //
    MallocFunc malloc_;
    FreeFunc free_;
    DeviceType device_type_;
    int device_index_;
public:
    CUDAAllocator() = delete;

    CUDAAllocator(DeviceType device, int index, MallocFunc malloc, FreeFunc free)
        : small_blocks_(block_comparator), large_blocks_(block_comparator)
    {
        malloc_ = malloc;
        free_ = free;
        device_type_ = device;
        device_index_ = index;
    }

    DataPtr do_split(Block *block, int64_t size)
    {
        std::lock_guard<std::recursive_mutex> l(mutex_);

        HMP_REQUIRE(block, "CUDAAllocator: Internal error");

        //split if possible
        auto &pool = *block->pool;
        Block *remaining = nullptr;
        if (should_split(block, size)){
            remaining = block;

            block = new Block(device_index_, size, block->ptr, block->pool);
            block->prev = remaining->prev;
            if (block->prev)
            {
                block->prev->next = block;
            }
            block->next = remaining;

            remaining->prev = block;
            remaining->ptr = static_cast<char *>(remaining->ptr) + block->size;
            remaining->size -= block->size;

            pool.insert(remaining);
        }

        auto stream = reinterpret_cast<cudaStream_t>(
            current_stream(kCUDA).value().handle());
        block->streams.insert(stream); //FIXME: take care of ptr used in multiple streams, now we only support current stream
        block->allocated = true;
        alloced_[block->ptr] = block;
        auto dptr = DataPtr(
            block->ptr, [=](void *ptr)
            { this->free(ptr); },
            Device(device_type_, device_index_));

        update_stat(stats_.active, block->size);
        update_stat(stats_.segment, 1);

        return dptr;
    }

    DataPtr alloc(int64_t size) override
    {
        process_events();

        //try alloc in pool
        Block *block = nullptr;
        size = round_size(size);
        auto& pool = get_pool(size);
        {
            std::lock_guard<std::recursive_mutex> l(mutex_);

            Block search_key(size);
            auto find_free_block = [&](){
                Block *block = nullptr;
                auto it = pool.lower_bound(&search_key);
                if(it != pool.end()){
                    block = *it;
                    pool.erase(it);
                }
                return block;
            };

            block = find_free_block();
            if(block != nullptr){
                return do_split(block, size);
            }
        }

        // try alloc by system allocator
        auto alloc_size = get_allocation_size(size);
        void *ptr = nullptr;
        auto rc = cuda_malloc_with_retry(&ptr, alloc_size);
        HMP_CUDA_CHECK(rc);

        block = new Block(device_index_, alloc_size, ptr, &pool);
        return do_split(block, size);
    }

    void free(void *ptr)
    {
        std::lock_guard<std::recursive_mutex> l(mutex_);
        auto it = alloced_.find(ptr);
        HMP_REQUIRE(it != alloced_.end(), "CUDAAllocator: free unknown ptr!!");

        //
        for(auto sit = it->second->streams.begin(); sit != it->second->streams.end(); ++sit){
            cudaEvent_t event;
            HMP_CUDA_CHECK(cudaEventCreate(&event));
            cudaEventRecord(event, *sit);
            freed_.push_back(std::make_pair(event, it->second));
            it->second->event_count += 1;
        }
        alloced_.erase(it);
        it->second->allocated = false;
        it->second->streams.clear();

        update_stat(stats_.active, -it->second->size);

        if(it->second->event_count == 0){
            free_block(it->second);
        }
        else{
            update_stat(stats_.inactive, it->second->size);
        }

        process_events();
    }

    void free_block(Block *block)
    {
        HMP_REQUIRE(block->event_count == 0, "CUDAAllocator: internal error");
        //
        std::lock_guard<std::recursive_mutex> l(mutex_);
        auto &pool = *block->pool;

        //merge block aggressively
        while(try_merge_blocks(block, block->prev, pool) > 0);
        while(try_merge_blocks(block, block->next, pool) > 0);

        pool.insert(block);

        update_stat(stats_.segment, -1);
    }

    void process_events()
    {
        std::lock_guard<std::recursive_mutex> l(mutex_);

        while(!freed_.empty()){
            cudaEvent_t event = freed_.front().first;
            auto block = freed_.front().second;

            auto err = cudaEventQuery(event);
            if(err == cudaErrorNotReady){
                cudaGetLastError();
                break;
            }
            else{
                HMP_CUDA_CHECK(err);

                HMP_CUDA_CHECK(cudaEventDestroy(event));
                block->event_count -= 1;
                if(block->event_count == 0){
                    auto block_size = block->size;
                    free_block(block);
                    update_stat(stats_.inactive, -block_size);
                }

                freed_.pop_front();
            }
        }
    }

    DeviceMemoryStats stats()
    {
        std::lock_guard<std::recursive_mutex> l(mutex_);
        process_events();
        return stats_;
    }

    //
    cudaError_t cuda_malloc_with_retry(void **ptr, size_t size)
    {
        auto rc = malloc_(ptr, size);
        if(rc != cudaSuccess){
            cudaGetLastError();  // reset the last CUDA error
            {
                std::lock_guard<std::recursive_mutex> l(mutex_);
                free_blocks(small_blocks_);
                free_blocks(large_blocks_);
            }

            rc = malloc_(ptr, size);
        }

        return rc;
    }

    void free_blocks(BlockPool &blocks)
    {
        //free all non-split cached blocks
        auto it = blocks.begin();
        while(it != blocks.end()){
            auto block = *it;
            if(block->is_split()){
                HMP_CUDA_CHECK(free_(block->ptr));
                
                auto cur = it;
                ++it;
                blocks.erase(cur);
                delete block;
            }
            else{
                ++it;
            }
        }
    }


    size_t try_merge_blocks(Block *dst, Block *src, BlockPool &pool)
    {
        if (!src || src->allocated || src->event_count > 0){
            return 0;
        }

        HMP_REQUIRE(dst->is_split() && src->is_split(), "CUDAAllocator: internal error");

        if (dst->prev == src){
            dst->ptr = src->ptr;
            dst->prev = src->prev;
            if (dst->prev){
                dst->prev->next = dst;
            }
        }
        else{
            dst->next = src->next;
            if (dst->next){
                dst->next->prev = dst;
            }
        }

        const size_t subsumed_size = src->size;
        dst->size += subsumed_size;
        pool.erase(src);
        delete src;

        return subsumed_size;
    }

    bool should_split(const Block *block, size_t size)
    {
        size_t remaining = block->size - size;
        if (block->pool == &small_blocks_){
            return remaining >= kMinBlockSize;
        }
        else if (block->pool == &large_blocks_){
            return remaining > kSmallSize;
        }
        else{
            HMP_REQUIRE(false, "Internal error");
        }
    }

    BlockPool &get_pool(size_t size)
    {
        if (size <= kSmallSize){
            return small_blocks_;
        }
        else{
            return large_blocks_;
        }
    }

    size_t round_size(size_t size)
    {
        if (size < kMinBlockSize){
            return kMinBlockSize;
        }
        else{
            return kMinBlockSize * ((size + kMinBlockSize - 1) / kMinBlockSize);
        }
    }

    size_t get_allocation_size(size_t size)
    {
        if (size <= kSmallSize){
            return kSmallBuffer;
        }
        else if (size < kMinLargeAlloc){
            return kLargeBuffer;
        }
        else{
            return kRoundLarge * ((size + kRoundLarge - 1) / kRoundLarge);
        }
    }
};


class CUDADeviceAllocator : public Allocator
{
    std::vector<std::unique_ptr<CUDAAllocator>> allocators_;
public:
    CUDADeviceAllocator()
    {
        int count = 0;
        try{
            HMP_CUDA_CHECK(cudaGetDeviceCount(&count));
        }
        catch(std::exception &e){
            HMP_WRN("cudaGetDeviceCount failed, {}", e.what());
        }

        allocators_.resize(count);
        for(int i = 0; i < count; ++i){
            allocators_[i] = std::unique_ptr<CUDAAllocator>(
                new CUDAAllocator(kCUDA, i, &cudaMalloc, &cudaFree));
        }
    }

    DataPtr alloc(int64_t size) override
    {
        int device;
        HMP_CUDA_CHECK(cudaGetDevice(&device));
        HMP_REQUIRE(device < allocators_.size(), 
            "device index {} is out of range {}", device, allocators_.size());

        return allocators_.at(device)->alloc(size);
    }

    DeviceMemoryStats stats(int device)
    {
        HMP_REQUIRE(device < allocators_.size(), 
            "device index {} is out of range {}", device, allocators_.size());
        return allocators_.at(device)->stats();
    }
};


static CUDADeviceAllocator sDefaultCUDAAllocator;
static CUDAAllocator sDefaultCPUAllocator(kCPU, 0, &cudaMallocHost, &cudaFreeHost);


} //namespace 

HMP_REGISTER_ALLOCATOR(kCUDA, &sDefaultCUDAAllocator, 0);
HMP_REGISTER_ALLOCATOR(kCPU, &sDefaultCPUAllocator, 1); //pinned


DeviceMemoryStats device_memory_stats(int device)
{
    return sDefaultCUDAAllocator.stats(device);
}

DeviceMemoryStats host_memory_stats()
{
    return sDefaultCPUAllocator.stats();
}


}} //namespace