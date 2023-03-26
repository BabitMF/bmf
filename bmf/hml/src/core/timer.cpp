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
#include <hmp/format.h>
#include <chrono>

namespace hmp{

namespace impl{

static TimerManager *sTimerManagers[static_cast<int>(DeviceType::NumDeviceTypes)];

void registerTimerManager(DeviceType dtype, TimerManager *tm)
{
    //as it only init before main, so no lock is needed
    sTimerManagers[static_cast<int>(dtype)] = tm; 
}

} //namespace impl


std::string stringfy(const Timer &timer)
{
    return fmt::format("Timer({}, {})", timer.device(), timer.is_stopped());
}

Timer create_timer(DeviceType dtype)
{
    auto tm = impl::sTimerManagers[static_cast<int>(dtype)];
    HMP_REQUIRE(tm, "Timer on device type {} is not supported", dtype);
    return tm->create();
}


//////
namespace {

const static Device sCPUDevice(kCPU);

class CPUTimer : public TimerInterface
{
    using TimePoint = decltype(std::chrono::high_resolution_clock::now()); 
    TimePoint begin_, end_;
    int state_ = -1; // -1 - not inited, 0 - stopped, 1 - started
public:
    void start() override 
    {
        begin_ = std::chrono::high_resolution_clock::now();
        state_ = 1;
    }

    void stop() override
    {
        HMP_REQUIRE(state_ == 1, "CPUTimer is not started");
        end_ = std::chrono::high_resolution_clock::now();
        state_ = 0;
    }

    double elapsed() override
    {
        TimePoint end = end_;
        if(state_ != 0){
            HMP_REQUIRE(state_ == 1, "CPUTimer is not inited");
            end = std::chrono::high_resolution_clock::now();
        }
        return std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin_).count()/1e9;
    }

    bool is_stopped() const override
    {
        return state_ == 0 || state_ == -1;
    }

    const Device& device() const override
    {
        return sCPUDevice;
    }
};

class CPUTimerManager : public impl::TimerManager
{
public:
    RefPtr<TimerInterface> create() override
    {
        return makeRefPtr<CPUTimer>();
    }
};


static CPUTimerManager scpuTimerManager;
HMP_REGISTER_TIMER_MANAGER(kCPU, &scpuTimerManager);

} //namespace

} //namespace hmp
