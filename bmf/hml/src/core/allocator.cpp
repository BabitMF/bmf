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

#include <hmp/core/logging.h>
#include <hmp/core/allocator.h>
#include <hmp/core/scalar_type.h>

namespace hmp{

void dummyDeleter(void*)
{
    //auto v = sizeof_scalar_type(kI8);
}


namespace {

class CPUAllocator : public Allocator
{
public:
    DataPtr alloc(int64_t size) override
    {
        //TODO: alignment support
        auto ptr = malloc(size);
        HMP_REQUIRE(ptr, "CPU out of memory");
        return DataPtr(ptr, free, kCPU);
    }
};

static CPUAllocator sDefaultCPUAllocator;

//+1(Pinned CPU allocator)
const static int sNumberDeviceTypes = static_cast<int>(DeviceType::NumDeviceTypes); 
static Allocator* sAllocators[sNumberDeviceTypes + 1];

} // namespace

HMP_DECLARE_ALLOCATOR(kCPU, 0);
HMP_DECLARE_ALLOCATOR(kCPU, 1);
HMP_DECLARE_ALLOCATOR(kCUDA, 0);


HMP_API void set_allocator(DeviceType device, Allocator *allocator, unsigned flags)
{
    HMP_REQUIRE(device < DeviceType::NumDeviceTypes, "invalid device type {}", device);
    if(device == kCPU && (flags & static_cast<unsigned>(AllocatorFlags::Pinned))){
        sAllocators[sNumberDeviceTypes] = allocator;
    }
    else{
        sAllocators[static_cast<int>(device)] = allocator;
    }
}
HMP_API Allocator *get_allocator(DeviceType device, unsigned flags)
{
#ifndef HMP_BUILD_SHARED
    HMP_IMPORT_ALLOCATOR(kCPU, 0);
#ifdef HMP_ENABLE_CUDA
    HMP_IMPORT_ALLOCATOR(kCPU, 1);
    HMP_IMPORT_ALLOCATOR(kCUDA, 0);
#endif
#endif

    //
    HMP_REQUIRE(device < DeviceType::NumDeviceTypes, "invalid device type {}", device);
    if(device == kCPU && (flags & static_cast<unsigned>(AllocatorFlags::Pinned))){
        return sAllocators[sNumberDeviceTypes];
    }
    else{
        return sAllocators[static_cast<int>(device)];
    }
}


HMP_REGISTER_ALLOCATOR(kCPU, &sDefaultCPUAllocator, 0);

} //namespace hmp