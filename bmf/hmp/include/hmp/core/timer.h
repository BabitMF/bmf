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

#include <hmp/core/device.h>
#include <hmp/core/ref_ptr.h>

namespace hmp {

class HMP_API TimerInterface : public RefObject {
  public:
    virtual void start() = 0;
    virtual void stop() = 0;
    virtual double elapsed() = 0;
    virtual bool is_stopped() const = 0;
    virtual const Device &device() const = 0;
};

class HMP_API Timer {
    RefPtr<TimerInterface> self_;

    Timer(RefPtr<TimerInterface> self) : self_(self) {}

    friend HMP_API Timer create_timer(DeviceType device_type);

  public:
    Timer(const Timer &) = default;
    Timer(Timer &&) = default;

    Timer &start() {
        self_->start();
        return *this;
    }
    Timer &stop() {
        self_->stop();
        return *this;
    }
    double elapsed() const { return self_->elapsed(); }
    bool is_stopped() const { return self_->is_stopped(); }
    const Device &device() const { return self_->device(); }
};

HMP_API std::string stringfy(const Timer &timer);
HMP_API Timer create_timer(DeviceType device_type = kCPU);

namespace impl {

struct HMP_API TimerManager {
    virtual RefPtr<TimerInterface> create() = 0;
};

HMP_API void registerTimerManager(DeviceType dtype, TimerManager *tm);

#define HMP_REGISTER_TIMER_MANAGER(dtype, tm)                                  \
    namespace {                                                                \
    static Register<DeviceType, ::hmp::impl::TimerManager *>                   \
        HMP_UNIQUE_NAME(s##timer##Manager)(::hmp::impl::registerTimerManager,  \
                                           dtype, tm);                         \
    }
} // namespace impl
}; // namespace hmp
