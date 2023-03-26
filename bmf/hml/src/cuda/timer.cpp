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
#include <hmp/core/timer.h>
#include <hmp/cuda/event.h>
#include <cuda_runtime.h>

namespace hmp{
namespace cuda{

namespace {

class CUDATimer : public TimerInterface
{
    Event begin_, end_;
    int state_ = -1; // -1 - not inited, 0 - stopped, 1 - started
    Device device_;
public:
    CUDATimer()
        : begin_(true), end_(true)
    {
        int index;
        HMP_CUDA_CHECK(cudaGetDevice(&index));
        device_ = Device(kCUDA, index); //FIXME: event may in different devices
    }

    void start() override 
    {
        begin_.record();
        state_ = 1;
    }

    void stop() override
    {
        HMP_REQUIRE(state_ == 1, "CUDATimer is not started");
        end_.record();
        state_ = 0;
    }

    double elapsed() override
    {
        HMP_REQUIRE(state_ == 0, "CUDATimer is not stopped");
        return begin_.elapsed(end_)/1e3;
    }

    bool is_stopped() const override
    {
        return state_ == 0 || state_ == -1;
    }

    const Device& device() const override
    {
        return device_;
    }
};

class CUDATimerManager : public impl::TimerManager
{
public:
    RefPtr<TimerInterface> create() override
    {
        return makeRefPtr<CUDATimer>();
    }
};


static CUDATimerManager scudaTimerManager;
HMP_REGISTER_TIMER_MANAGER(kCUDA, &scudaTimerManager);

} //namespace


}} //namespace