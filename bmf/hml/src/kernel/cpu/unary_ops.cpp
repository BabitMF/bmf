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

#include <kernel/unary_ops.h>
#include <kernel/cpu/kernel_utils.h>

namespace hmp{
namespace kernel{
namespace {

Tensor& round_cpu(Tensor &out, const Tensor &in)
{
    HMP_DISPATCH_FLOATING_POINT_TYPES_AND_HALF(in.scalar_type(), "round_cpu", [&](){
        cpu::uop_kernel<scalar_t, scalar_t>(out, in, [&](scalar_t v){
            return std::round(v);
        });
    });
    return out;
}

Tensor& ceil_cpu(Tensor &out, const Tensor &in)
{
    HMP_DISPATCH_FLOATING_POINT_TYPES_AND_HALF(in.scalar_type(), "ceil_cpu", [&](){
        cpu::uop_kernel<scalar_t, scalar_t>(out, in, [&](scalar_t v){
            return std::ceil(v);
        });
    });
    return out;
}

Tensor& floor_cpu(Tensor &out, const Tensor &in)
{
    HMP_DISPATCH_FLOATING_POINT_TYPES_AND_HALF(in.scalar_type(), "floor_cpu", [&](){
        cpu::uop_kernel<scalar_t, scalar_t>(out, in, [&](scalar_t v){
            return std::floor(v);
        });
    });
    return out;
}


Tensor& abs_cpu(Tensor &out, const Tensor &in)
{
    HMP_DISPATCH_ALL_TYPES_AND_HALF(in.scalar_type(), "abs_cpu", [&](){
        cpu::uop_kernel<scalar_t, scalar_t>(out, in, [&](scalar_t v){
            return std::abs(v);
        });
    });
    return out;
}


Tensor& minus_cpu(Tensor &out, const Tensor &in)
{
    HMP_DISPATCH_ALL_TYPES_AND_HALF(in.scalar_type(), "minus_cpu", [&](){
        cpu::uop_kernel<scalar_t, scalar_t>(out, in, [&](scalar_t v){
            return -v;
        });
    });
    return out;
}


Tensor& clip_cpu(Tensor &out, const Tensor &in, const Scalar &min, const Scalar &max)
{
    HMP_DISPATCH_ALL_TYPES_AND_HALF(in.scalar_type(), "clip_cpu", [&](){
        auto min_v = min.to<scalar_t>();
        auto max_v = max.to<scalar_t>();

        HMP_REQUIRE(min_v <= max_v,
             "clip_cpu: expect min <= max, got min={}, max={}", min_v, max_v);

        cpu::uop_kernel<scalar_t, scalar_t>(out, in, [&](scalar_t v){
            return v < min_v ? min_v : (v > max_v ? max_v : v);
        });
    });
    return out;
}


HMP_DEVICE_DISPATCH(kCPU, round_stub, &round_cpu)
HMP_DEVICE_DISPATCH(kCPU, ceil_stub, &ceil_cpu)
HMP_DEVICE_DISPATCH(kCPU, floor_stub, &floor_cpu)
HMP_DEVICE_DISPATCH(kCPU, abs_stub, &abs_cpu)
HMP_DEVICE_DISPATCH(kCPU, minus_stub, &minus_cpu)
HMP_DEVICE_DISPATCH(kCPU, clip_stub, &clip_cpu)


}}} //