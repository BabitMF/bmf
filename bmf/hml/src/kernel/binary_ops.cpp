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
#include <kernel/kernel_utils.h>

namespace hmp{
namespace kernel{

HMP_DEFINE_DISPATCH_STUB(mul_stub)
HMP_DEFINE_DISPATCH_STUB(mul_scalar_stub)
HMP_DEFINE_DISPATCH_STUB(add_stub)
HMP_DEFINE_DISPATCH_STUB(add_scalar_stub)
HMP_DEFINE_DISPATCH_STUB(sub_stub)
HMP_DEFINE_DISPATCH_STUB(sub_scalar_stub)
HMP_DEFINE_DISPATCH_STUB(sub_scalar_stub2)
HMP_DEFINE_DISPATCH_STUB(div_stub)
HMP_DEFINE_DISPATCH_STUB(div_scalar_stub)
HMP_DEFINE_DISPATCH_STUB(div_scalar_stub2)


#define WRAP_BINARY_OP(op)\
    Tensor& op(Tensor &out, const Tensor &ina, const Tensor &inb) \
    { \
        checkShape({out, ina, inb}, out.shape(), #op); \
        checkDevice({out, ina, inb}, out.device(), #op); \
        return op##_stub(out.device_type(), out, ina, inb); \
    }   \
    Tensor& op(Tensor &out, const Tensor &ina, const Scalar &inb)  \
    { \
        checkShape({out, ina}, out.shape(), #op); \
        checkDevice({out, ina}, out.device(), #op); \
        return op##_scalar_stub(out.device_type(), out, ina, inb); \
    }


#define WRAP_BINARY_OP2(op)\
    Tensor& op(Tensor &out, const Tensor &ina, const Tensor &inb) \
    { \
        checkShape({out, ina, inb}, out.shape(), #op); \
        checkDevice({out, ina, inb}, out.device(), #op); \
        return op##_stub(out.device_type(), out, ina, inb); \
    }   \
    Tensor& op(Tensor &out, const Tensor &ina, const Scalar &inb)  \
    { \
        checkShape({out, ina}, out.shape(), #op); \
        checkDevice({out, ina}, out.device(), #op); \
        return op##_scalar_stub(out.device_type(), out, ina, inb); \
    } \
    Tensor& op(Tensor &out, const Scalar &ina, const Tensor &inb)  \
    { \
        checkShape({out, inb}, out.shape(), #op); \
        checkDevice({out, inb}, out.device(), #op); \
        return op##_scalar_stub2(out.device_type(), out, ina, inb); \
    }

WRAP_BINARY_OP(mul)
WRAP_BINARY_OP(add)
WRAP_BINARY_OP2(sub)
WRAP_BINARY_OP2(div)


}} //namespace hmp::kernel