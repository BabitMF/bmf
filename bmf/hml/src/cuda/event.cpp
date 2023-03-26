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
#include <hmp/cuda/event.h>
#include <cuda_runtime.h>

namespace hmp{
namespace cuda{

Event::Event()
    : Event(false, true, false)
{
}

Event::Event(Event &&other)
{
    event_ = other.event_;
    flags_ = other.flags_;
    is_created_ = other.is_created_;
    device_index_ = other.device_index_;

    other.device_index_ = -1;
    other.event_ = 0;
    other.is_created_ = false;
}


Event::Event(bool enable_timing, bool blocking, bool interprocess)
{
    event_ = 0;
    is_created_ = false;
    device_index_ = -1;

    flags_ = 0;
    if (!enable_timing){
        flags_ |= (unsigned int)cudaEventDisableTiming;
    }
    if (blocking){
        flags_ |= (unsigned int)cudaEventBlockingSync;
    }
    if (interprocess){
        flags_ |= (unsigned int)cudaEventInterprocess;
    }
}

Event::~Event()
{
    if(is_created_){
        HMP_CUDA_CHECK_WRN(cudaEventDestroy(reinterpret_cast<cudaEvent_t>(event_)));
    }
}

void Event::record(const optional<Stream> &stream_)
{
    auto stream = stream_.value_or(current_stream(kCUDA).value());
    if(!is_created_){
        cudaEvent_t event;
        HMP_CUDA_CHECK(cudaEventCreateWithFlags(&event, flags_));
        is_created_ = true;
        HMP_CUDA_CHECK(cudaGetDevice(&device_index_));
        event_ = event;
    }

    HMP_REQUIRE(device_index_ == stream.device().index(),
        "Event is create on {} dose not match recording stream's device {}",
        device_index_, stream.device().index());
    HMP_CUDA_CHECK(cudaEventRecord(
        reinterpret_cast<cudaEvent_t>(event_), 
        reinterpret_cast<cudaStream_t>(stream.handle())));
}

void Event::block(const optional<Stream> &stream_)
{
    auto stream = stream_.value_or(current_stream(kCUDA).value());
    if(is_created_){
        auto err = cudaStreamWaitEvent(
                    reinterpret_cast<cudaStream_t>(stream.handle()),
                    reinterpret_cast<cudaEvent_t>(event_), 0);
        HMP_CUDA_CHECK(err);
    }
}


bool Event::query() const
{
    if(!is_created_){
        return true;
    }

    auto err = cudaEventQuery(reinterpret_cast<cudaEvent_t>(event_));
    if(err == cudaSuccess){
        return true;
    }
    else if(err != cudaErrorNotReady){
        HMP_CUDA_CHECK(err);
    }
    return false;
}

void Event::synchronize() const
{
    if(is_created_){
        HMP_CUDA_CHECK(cudaEventSynchronize(reinterpret_cast<cudaEvent_t>(event_)));
    }
}

float Event::elapsed(const Event &other)
{
    HMP_REQUIRE(is_created() && other.is_created(),
         "Event: Both events need be created");
    float ms = 0;
    HMP_CUDA_CHECK(cudaEventElapsedTime(&ms,
         reinterpret_cast<cudaEvent_t>(event_), 
         reinterpret_cast<cudaEvent_t>(other.event_)));
    return ms;
}


}} //namespace