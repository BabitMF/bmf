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

#include <hmp/core/stream.h>
#include <hmp/cuda/macros.h>
#include <cuda_runtime.h>

#include <mutex>
#include <thread>
#include <deque>

namespace hmp{
namespace cuda{

class CUDAStreamCache
{
    std::mutex mutex_;
    std::deque<cudaStream_t> streamCache_[MaxDevices];
public:
    cudaStream_t create(int device)
    {
        std::lock_guard<std::mutex> l(mutex_);
        HMP_REQUIRE(device < MaxDevices,
             "CUDAStreamCache: device index({}) is out of range {}", device, MaxDevices);

        cudaStream_t stream = 0;
        auto &cache = streamCache_[device];
        if(!cache.empty()){
            stream = cache.back();
            cache.pop_back();
        }
        else{
            int oldDevice;
            HMP_CUDA_CHECK(cudaGetDevice(&oldDevice));
            HMP_CUDA_CHECK(cudaSetDevice(device));
            HMP_CUDA_CHECK(cudaStreamCreate(&stream));
            HMP_CUDA_CHECK(cudaSetDevice(oldDevice));
        }

        return stream;
    }

    void destroy(cudaStream_t stream, int device)
    {
        HMP_REQUIRE(device < MaxDevices,
             "CUDAStreamCache: device index({}) is out of range {}", device, MaxDevices);
        HMP_CUDA_CHECK(cudaStreamSynchronize(stream));
        //
        std::lock_guard<std::mutex> l(mutex_);
        streamCache_[device].push_back(stream);
    }
};


static CUDAStreamCache &streamCache()
{
    static CUDAStreamCache scache;
    return scache;
}



class CUDAStream : public StreamInterface
{
    Device device_;
    cudaStream_t stream_;
    bool own_;
public:
    CUDAStream() : own_(false), stream_(0) //default stream
    {
        auto device = current_device(kCUDA);
        HMP_REQUIRE(device, "No CUDA device have been selected");
        device_ = device.value();
    }

    CUDAStream(cudaStream_t stream, bool own) : own_(own), stream_(stream) //default stream
    {
        auto device = current_device(kCUDA);
        HMP_REQUIRE(device, "No CUDA device have been selected");
        device_ = device.value();
    }

    CUDAStream(uint64_t flags)
        : CUDAStream()
    {
        stream_ = streamCache().create(device_.index());
        own_ = true;
    }

    ~CUDAStream()
    {
        //do not destroy stream, as it may used in allocator
        if(stream_ != 0 && own_){
            streamCache().destroy(stream_, device_.index());
        }
    }

    const Device &device() const override
    {
        return device_;
    }

    StreamHandle handle() const override
    {
        static_assert(sizeof(StreamHandle) >= sizeof(cudaStream_t), "invalid size of cudaStream_t");
        return reinterpret_cast<StreamHandle>(stream_);
    }

    bool query() override
    {
        auto rc = cudaStreamQuery(stream_);
        return rc == cudaSuccess;
    }

    virtual void synchronize() override
    {
        HMP_CUDA_CHECK(cudaStreamSynchronize(stream_));
    }
}; 


HMP_API Stream wrap_stream(StreamHandle stream, bool own)
{
    return Stream(makeRefPtr<CUDAStream>(cudaStream_t(stream), own));
}

//
static thread_local RefPtr<CUDAStream> sCurrentStream; 

class CUDAStreamManager : public impl::StreamManager
{
public:
    void setCurrent(const Stream& stream) override
    {
        auto ref = stream.unsafeGet();
        auto cudaStream = dynamic_cast<CUDAStream*>(ref.get());
        HMP_REQUIRE(cudaStream, "Invalid CUDA stream");

        sCurrentStream = ref.cast<CUDAStream>();
    }

    optional<Stream> getCurrent() const override
    {
        if(!sCurrentStream){
            return Stream(makeRefPtr<CUDAStream>()); //get default stream by default
        }
        else{
            return Stream(sCurrentStream);
        }
    }

    Stream create(uint64_t flags = 0) override
    {
        return Stream(makeRefPtr<CUDAStream>(flags));
    }

};

static CUDAStreamManager sCUDAStreamManager;

HMP_REGISTER_STREAM_MANAGER(kCUDA, &sCUDAStreamManager);

}} //namesapce