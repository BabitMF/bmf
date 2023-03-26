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

#include <string_view>
#include <mutex>
#include <fmt/format.h>
#include <hmp/core/logging.h>
#include <hmp/core/device.h>

namespace hmp{


HMP_DECLARE_DEVICE(kCPU);
#ifdef HMP_ENABLE_CUDA
HMP_DECLARE_DEVICE(kCUDA);
#endif

Device::Device(Type type, Index index)
    : type_(type), index_(index)
{
    HMP_REQUIRE(index >= 0, "invalid device index {} of {}", index, type);

#ifndef HMP_BUILD_SHARED
    HMP_IMPORT_DEVICE(kCPU);
#ifdef HMP_ENABLE_CUDA
    HMP_IMPORT_DEVICE(kCUDA);
#endif
#endif

}


Device::Device(const std::string &devstr)
{
    auto cpos = devstr.find(":");
    std::string_view dstr{devstr};
    int index = 0;
    if(cpos != std::string::npos){
        dstr = std::string_view(devstr.c_str(), cpos);
        const char *start = devstr.c_str() + cpos + 1;
        char *end = nullptr;
        index = strtol(start, &end, 10);
        HMP_REQUIRE(start < end, "invalid device index in devstr '{}'", devstr);
    }

    if(dstr == "cpu"){
        type_ = kCPU;
    }
    else if(dstr == "cuda"){
        type_ = kCUDA;
    }
    else{
        HMP_REQUIRE(false, "invalid device string '{}'", devstr);
    }

    //
    auto count = device_count(type_);
    HMP_REQUIRE(index < count,
         "device index({}) is out of range({})", index, count);
    index_ = index;
}

bool Device::operator==(const Device &other) const
{
    return type_ == other.type() && index_ == other.index();
}



std::string stringfy(const Device &device)
{
    if(device.type() == kCPU){
        return "cpu";
    }
    else if(device.type() == kCUDA){
        return fmt::format("cuda:{}", device.index());
    }
    else{
        return "InvalidDevice";
    }
}


/////////

namespace impl{

static DeviceManager *sDeviceManagers[static_cast<int>(DeviceType::NumDeviceTypes)];

void registerDeviceManager(DeviceType dtype, DeviceManager *dm)
{
    //as it only init before main, so no lock is needed
    sDeviceManagers[static_cast<int>(dtype)] = dm; 
}

////
class CPUDeviceManager : public DeviceManager
{
    static Device cpuDevice_;
public:
    void setCurrent(const Device &) override
    {
    }
    optional<Device> getCurrent() const override
    {
        return cpuDevice_;
    }

    int64_t count() const override
    {
        return 1;
    }
};
Device CPUDeviceManager::cpuDevice_{};

static CPUDeviceManager sCPUDeviceManager;
HMP_REGISTER_DEVICE_MANAGER(kCPU, &sCPUDeviceManager);

} //namespace impl


optional<Device> current_device(DeviceType dtype)
{
    auto dm = impl::sDeviceManagers[static_cast<int>(dtype)];
    HMP_REQUIRE(dm, "Device type {} is not supported", dtype);
    return dm->getCurrent();
}

void set_current_device(const Device &device)
{
    auto dtype = device.type();
    auto dm = impl::sDeviceManagers[static_cast<int>(dtype)];
    HMP_REQUIRE(dm, "Device type {} is not supported", dtype);
    dm->setCurrent(device);
}

int64_t device_count(DeviceType dtype)
{
    auto dm = impl::sDeviceManagers[static_cast<int>(dtype)];
    if(dm){
        return dm->count();
    }
    else{
        return 0;
    }
}

DeviceGuard::DeviceGuard(const Device &device)
{
    auto current = current_device(device.type());
    if (current != device){
        set_current_device(device);
        origin_ = current;
    }
}

DeviceGuard::DeviceGuard(DeviceGuard &&other)
{
    origin_ = other.origin_;
    other.origin_.reset();
}

DeviceGuard::~DeviceGuard()
{
    if(origin_){
        set_current_device(origin_.value());
    }
}

//



} //namespace hmp