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

#include <mutex>
#include <hmp/cuda/macros.h>
#include <hmp/cuda/device.h>
#include <hmp/core/device.h>
#include <cuda_runtime.h>
#include <cuda.h>

namespace hmp{
namespace cuda{


class CUDADeviceManager : public impl::DeviceManager
{
public:
    CUDADeviceManager()
    {
        initContext();
    }
    static void initContext()
    {
        cuInit(0);
        CUdevice device;
        const unsigned int flags = CU_CTX_SCHED_BLOCKING_SYNC;
        int count;
        HMP_CUDA_CHECK(cudaGetDeviceCount(&count));
        for (int idx = 0; idx < count; idx++) {
            auto ret = cuDeviceGet(&device, idx);
            HMP_REQUIRE(ret == CUDA_SUCCESS, "get CUdevice {} failed={}", idx, ret);
            ret = cuDevicePrimaryCtxSetFlags(device, flags);
            HMP_REQUIRE(ret == CUDA_SUCCESS, "set device {} primary ctx flags failed={}", idx, ret);
        }
        return;
    }
    void setCurrent(const Device &dev) override
    {
        HMP_CUDA_CHECK(cudaSetDevice(dev.index()));

    }
    optional<Device> getCurrent() const override
    {
        int index = 0;
        HMP_CUDA_CHECK(cudaGetDevice(&index));
        return Device(kCUDA, index);
    }

    int64_t count() const
    {
        int count = 0;
        HMP_CUDA_CHECK(cudaGetDeviceCount(&count));
        return count;
    }
};

static CUDADeviceManager sCUDADeviceManager;
HMP_REGISTER_DEVICE_MANAGER(kCUDA, &sCUDADeviceManager);

static const cudaDeviceProp &get_device_prop(int device)
{
    static cudaDeviceProp sProps[MaxDevices];
    static cudaDeviceProp* sPProps[MaxDevices];
    static std::mutex sPropsLock;

    HMP_REQUIRE(device < MaxDevices, 
        "{} is exceed cuda::MaxDevices limitation {}", device, MaxDevices);

    if(sPProps[device] == nullptr){
        std::lock_guard l(sPropsLock);
        if(sPProps[device] == nullptr){
            HMP_CUDA_CHECK(cudaGetDeviceProperties(sProps+device, device));
            sPProps[device] = sProps + device;
        }
    }

    return *sPProps[device];
}


int64_t DeviceProp::texture_pitch_alignment()
{
    int device;
    HMP_CUDA_CHECK(cudaGetDevice(&device));

    auto &prop = get_device_prop(device);
    return prop.texturePitchAlignment;
}




}} //namespace
