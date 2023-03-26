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
#include <kernel/binary_ops.h>
#include <kernel/cpu/kernel_utils.h>

namespace hmp{
namespace kernel{
namespace {

Tensor& mul_cpu(Tensor &out, const Tensor &ina, const Tensor &inb)
{
    HMP_DISPATCH_ALL_TYPES_AND_HALF(out.scalar_type(), "mul_cpu", [&](){
        cpu::bop_kernel<scalar_t, scalar_t, scalar_t>(out, ina, inb, 
            [&](scalar_t a, scalar_t b){
                return a * b;
        });
    });
    return out;
}


Tensor& mul_scalar_cpu(Tensor &out, const Tensor &ina, const Scalar &inb)
{
    HMP_DISPATCH_ALL_TYPES_AND_HALF(out.scalar_type(), "mul_scalar_cpu", [&](){
        auto b = inb.to<scalar_t>();
        cpu::uop_kernel<scalar_t, scalar_t>(out, ina,
            [&](scalar_t a){
                return a * b;
        });
    });
    return out;
}


Tensor& add_cpu(Tensor &out, const Tensor &ina, const Tensor &inb)
{
    HMP_DISPATCH_ALL_TYPES_AND_HALF(out.scalar_type(), "add_cpu", [&](){
        cpu::bop_kernel<scalar_t, scalar_t, scalar_t>(out, ina, inb, 
            [&](scalar_t a, scalar_t b){
                return a + b;
        });
    });
    return out;
}


Tensor& add_scalar_cpu(Tensor &out, const Tensor &ina, const Scalar &inb)
{
    HMP_DISPATCH_ALL_TYPES_AND_HALF(out.scalar_type(), "add_scalar_cpu", [&](){
        auto b = inb.to<scalar_t>();
        cpu::uop_kernel<scalar_t, scalar_t>(out, ina,
            [&](scalar_t a){
                return a + b;
        });
    });
    return out;
}


Tensor& sub_cpu(Tensor &out, const Tensor &ina, const Tensor &inb)
{
    HMP_DISPATCH_ALL_TYPES_AND_HALF(out.scalar_type(), "sub_cpu", [&](){
        cpu::bop_kernel<scalar_t, scalar_t, scalar_t>(out, ina, inb, 
            [&](scalar_t a, scalar_t b){
                return a - b;
        });
    });
    return out;
}


Tensor& sub_scalar_cpu(Tensor &out, const Tensor &ina, const Scalar &inb)
{
    HMP_DISPATCH_ALL_TYPES_AND_HALF(out.scalar_type(), "sub_scalar_cpu", [&](){
        auto b = inb.to<scalar_t>();
        cpu::uop_kernel<scalar_t, scalar_t>(out, ina,
            [&](scalar_t a){
                return a - b;
        });
    });
    return out;
}


Tensor& sub_scalar2_cpu(Tensor &out, const Scalar &ina, const Tensor &inb)
{
    HMP_DISPATCH_ALL_TYPES_AND_HALF(out.scalar_type(), "sub_scalar_cpu", [&](){
        auto a = ina.to<scalar_t>();
        cpu::uop_kernel<scalar_t, scalar_t>(out, inb,
            [&](scalar_t b){
                return a - b;
        });
    });
    return out;
}


Tensor& div_cpu(Tensor &out, const Tensor &ina, const Tensor &inb)
{
    HMP_DISPATCH_ALL_TYPES_AND_HALF(out.scalar_type(), "div_cpu", [&](){
        cpu::bop_kernel<scalar_t, scalar_t, scalar_t>(out, ina, inb, 
            [&](scalar_t a, scalar_t b){
                return a / b;
        });
    });
    return out;
}


Tensor& div_scalar_cpu(Tensor &out, const Tensor &ina, const Scalar &inb)
{
    HMP_DISPATCH_ALL_TYPES_AND_HALF(out.scalar_type(), "div_scalar_cpu", [&](){
        auto b = inb.to<scalar_t>();
        cpu::uop_kernel<scalar_t, scalar_t>(out, ina,
            [&](scalar_t a){
                return a / b;
        });
    });
    return out;
}


Tensor& div_scalar2_cpu(Tensor &out, const Scalar &ina, const Tensor &inb)
{
    HMP_DISPATCH_ALL_TYPES_AND_HALF(out.scalar_type(), "div_scalar_cpu", [&](){
        auto a = ina.to<scalar_t>();
        cpu::uop_kernel<scalar_t, scalar_t>(out, inb,
            [&](scalar_t b){
                return a / b;
        });
    });
    return out;
}



HMP_DEVICE_DISPATCH(kCPU, mul_stub, &mul_cpu)
HMP_DEVICE_DISPATCH(kCPU, mul_scalar_stub, &mul_scalar_cpu)
HMP_DEVICE_DISPATCH(kCPU, add_stub, &add_cpu)
HMP_DEVICE_DISPATCH(kCPU, add_scalar_stub, &add_scalar_cpu)
HMP_DEVICE_DISPATCH(kCPU, sub_stub, &sub_cpu)
HMP_DEVICE_DISPATCH(kCPU, sub_scalar_stub, &sub_scalar_cpu)
HMP_DEVICE_DISPATCH(kCPU, sub_scalar_stub2, &sub_scalar2_cpu)
HMP_DEVICE_DISPATCH(kCPU, div_stub, &div_cpu)
HMP_DEVICE_DISPATCH(kCPU, div_scalar_stub, &div_scalar_cpu)
HMP_DEVICE_DISPATCH(kCPU, div_scalar_stub2, &div_scalar2_cpu)


}}} //