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

#include <kernel/tensor_factory.h>
#include <kernel/cpu/kernel_utils.h>

namespace hmp{
namespace kernel{
namespace{

Tensor &fill_cpu_impl(Tensor &self, const Scalar &value)
{
    HMP_DISPATCH_ALL_TYPES_AND_HALF(self.scalar_type(), "fill_cpu", [&](){
        auto v = value.to<scalar_t>();

        cpu::gen_kernel<scalar_t>(self, [&](int64_t idx){
            return v;
        });
    });

    return self;
}


Tensor &arange_cpu_impl(Tensor &self, int64_t start, int64_t end, int64_t step)
{
    HMP_DISPATCH_ALL_TYPES(self.scalar_type(), "arange_cpu", [&](){
        cpu::gen_kernel<scalar_t>(self, [&](int64_t idx){
            return start + idx * step;
        });
    });

    return self;
}


Tensor &copy_cpu_impl(Tensor &self, const Tensor &other)
{
#ifndef HMP_ENABLE_MOBILE
    HMP_DISPATCH_ALL_TYPES_AND_HALF(self.scalar_type(), "copy_cpu", [&](){
        using oscalar_t = scalar_t;
        HMP_DISPATCH_ALL_TYPES_AND_HALF(other.scalar_type(), "copy_cpu", [&](){
            using iscalar_t = scalar_t;
            cpu::uop_kernel<oscalar_t, iscalar_t>(self, other,
                 [&](scalar_t v){
                    return cast<oscalar_t>(v);
            });
        });
    });
#else
    HMP_DISPATCH_ALL_TYPES_AND_HALF(self.scalar_type(), "copy_cpu", [&](){
        using oscalar_t = scalar_t;
        using iscalar_t = scalar_t;
        cpu::uop_kernel<oscalar_t, iscalar_t>(self, other,
            [&](scalar_t v){
                return cast<oscalar_t>(v);
        });
    });
#endif

    return self;
}


HMP_DEVICE_DISPATCH(kCPU, fill_stub, &fill_cpu_impl)
HMP_DEVICE_DISPATCH(kCPU, copy_stub, &copy_cpu_impl)
HMP_DEVICE_DISPATCH(kCPU, arange_stub, &arange_cpu_impl)

}}} //