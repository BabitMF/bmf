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
#include <kernel/kernel_utils.h>

namespace hmp{
namespace kernel{

HMP_DEFINE_DISPATCH_STUB(round_stub)
HMP_DEFINE_DISPATCH_STUB(ceil_stub)
HMP_DEFINE_DISPATCH_STUB(floor_stub)
HMP_DEFINE_DISPATCH_STUB(abs_stub)
HMP_DEFINE_DISPATCH_STUB(minus_stub)
HMP_DEFINE_DISPATCH_STUB(clip_stub)


#define WRAP_UNARY_OP(op) \
    Tensor& op(Tensor& out, const Tensor &input) \
    { \
        checkShape({out, input}, input.shape(), #op); \
        checkDevice({out, input}, input.device(), #op); \
        return op##_stub(input.device_type(), out, input); \
    }

WRAP_UNARY_OP(round)
WRAP_UNARY_OP(ceil)
WRAP_UNARY_OP(floor)
WRAP_UNARY_OP(abs)
WRAP_UNARY_OP(minus)



Tensor& clip(Tensor &out, const Tensor &input, const Scalar &min, const Scalar &max)
{
    checkShape({out, input}, input.shape(), "clip");
    checkDevice({out, input}, input.device(), "clip");

    return clip_stub(input.device_type(), out, input, min, max);
}



}} //namespace hmp::kernel