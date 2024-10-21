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

#include <string>
#include <stdint.h>
#include <hmp/core/optional.h>
#include <hmp/core/macros.h>

namespace hmp {

class HMP_API Device {
  public:
    enum class Type : int16_t {
        CPU = 0,
        CUDA = 1,

        //
        NumDeviceTypes
    };
    using Index = int16_t;

    Device() : Device(Type::CPU) {}
    Device(Type type, Index index = 0);
    Device(const std::string &devstr);

    //
    bool operator==(const Device &other) const;
    bool operator!=(const Device &other) const { return !(*this == other); }

    //
    Type type() const { return type_; }
    Index index() const { return index_; }

  private:
    Type type_;
    Index index_;
};

using DeviceType = Device::Type;
constexpr Device::Type kCPU = Device::Type::CPU;
constexpr Device::Type kCUDA = Device::Type::CUDA;

HMP_API std::string stringfy(const Device &device);

static inline std::string stringfy(const Device::Type &type) {
    if (type == kCPU) {
        return "kCPU";
    } else if (type == kCUDA) {
        return "kCUDA";
    } else {
        return "UnknownDeviceType";
    }
}

class HMP_API DeviceGuard {
    optional<Device> origin_;

  public:
    DeviceGuard() = delete;
    DeviceGuard(const DeviceGuard &) = delete;
    DeviceGuard(DeviceGuard &&);

    DeviceGuard(const Device &device);
    ~DeviceGuard();

    optional<Device> original() { return origin_; }
};

HMP_API int64_t device_count(DeviceType device_type);
HMP_API optional<Device> current_device(DeviceType device_type);
HMP_API void set_current_device(const Device &device);

namespace impl {

struct DeviceManager {
    virtual void setCurrent(const Device &) = 0;
    virtual optional<Device> getCurrent() const = 0;
    virtual int64_t count() const = 0;
};

HMP_API void registerDeviceManager(DeviceType dtype, DeviceManager *dm);

#define HMP_REGISTER_DEVICE_MANAGER(device, dm)                                \
    namespace {                                                                \
    static Register<DeviceType, ::hmp::impl::DeviceManager *>                  \
        __s##device##Manager(::hmp::impl::registerDeviceManager, device, dm);  \
    HMP_DEFINE_TAG(__s##device##Manager);                                      \
    }

#define HMP_DECLARE_DEVICE(device) HMP_DECLARE_TAG(__s##device##Manager)
#define HMP_IMPORT_DEVICE(device) HMP_IMPORT_TAG(__s##device##Manager)

} // namespace impl

} // namespace hmp