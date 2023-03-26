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
#include <hmp/tensor.h>
#include <kernel/dispatch_stub.h>

namespace hmp{
namespace kernel{

using unary_ops = Tensor&(*)(Tensor&, const Tensor&);

HMP_DECLARE_DISPATCH_STUB(round_stub, unary_ops)
HMP_DECLARE_DISPATCH_STUB(ceil_stub, unary_ops)
HMP_DECLARE_DISPATCH_STUB(floor_stub, unary_ops)
HMP_DECLARE_DISPATCH_STUB(abs_stub, unary_ops)
HMP_DECLARE_DISPATCH_STUB(minus_stub, unary_ops)
HMP_DECLARE_DISPATCH_STUB(clip_stub, Tensor&(*)(Tensor&, const Tensor&, const Scalar&, const Scalar&))


Tensor& round(Tensor &out, const Tensor &in);
Tensor& ceil(Tensor &out, const Tensor &in);
Tensor& floor(Tensor &out, const Tensor &in);
Tensor& abs(Tensor &out, const Tensor &in);
Tensor& minus(Tensor &out, const Tensor &in);
Tensor& clip(Tensor &out, const Tensor &in, const Scalar &min, const Scalar &max);


}} //namespace