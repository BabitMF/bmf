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

#include <memory>
#include <functional>
#include <stdint.h>
#include <hmp/core/macros.h>
#include <hmp/core/device.h>

namespace hmp {

HMP_API void dummyDeleter(void *);

class HMP_API DataPtr {
  public:
    using DeleterFnPtr = std::function<void(void *)>;

    DataPtr() : ptr_(nullptr, nullptr), device_(kCPU) {}
    DataPtr(void *data, DeleterFnPtr deleter, Device device)
        : ptr_(data, deleter), device_(device) {}
    DataPtr(void *data, Device device) : DataPtr(data, dummyDeleter, device) {}

    void *operator->() const { return ptr_.get(); }

    operator bool() const { return ptr_.get() != nullptr; }

    void release() { ptr_.release(); }

    void *get() const { return ptr_.get(); }

    const Device &device() const { return device_; }

  private:
    std::unique_ptr<void, DeleterFnPtr> ptr_;
    Device device_;
};

struct HMP_API Allocator {
    enum class Flags : uint8_t {
        Pinned = 0x1 // valid for kCPU, ignored by other device types
    };

    virtual DataPtr alloc(int64_t size) = 0;
};
using AllocatorFlags = Allocator::Flags;

HMP_API void set_allocator(DeviceType device, Allocator *allocator,
                           unsigned flags = 0);
HMP_API Allocator *get_allocator(DeviceType device, unsigned flags = 0);

#define HMP_REGISTER_ALLOCATOR(device, alloc, flags)                           \
    static Register<DeviceType, Allocator *, unsigned>                         \
        __s##device##flags##Allocator(set_allocator, device, alloc, flags);    \
    HMP_DEFINE_TAG(__s##device##flags##Allocator);

#define HMP_DECLARE_ALLOCATOR(device, flags)                                   \
    HMP_DECLARE_TAG(__s##device##flags##Allocator)
#define HMP_IMPORT_ALLOCATOR(device, flags)                                    \
    HMP_IMPORT_TAG(__s##device##flags##Allocator)

} // namespace hmp