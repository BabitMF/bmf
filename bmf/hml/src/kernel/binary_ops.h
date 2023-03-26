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

using binary_op = Tensor&(*)(Tensor&, const Tensor&, const Tensor&);
using binary_op_scalar = Tensor&(*)(Tensor&, const Tensor&, const Scalar&);
using binary_op_scalar2 = Tensor&(*)(Tensor&, const Scalar&, const Tensor&);

HMP_DECLARE_DISPATCH_STUB(mul_stub, binary_op)
HMP_DECLARE_DISPATCH_STUB(mul_scalar_stub, binary_op_scalar)
HMP_DECLARE_DISPATCH_STUB(add_stub, binary_op)
HMP_DECLARE_DISPATCH_STUB(add_scalar_stub, binary_op_scalar)
HMP_DECLARE_DISPATCH_STUB(sub_stub, binary_op)
HMP_DECLARE_DISPATCH_STUB(sub_scalar_stub, binary_op_scalar)
HMP_DECLARE_DISPATCH_STUB(sub_scalar_stub2, binary_op_scalar2)
HMP_DECLARE_DISPATCH_STUB(div_stub, binary_op)
HMP_DECLARE_DISPATCH_STUB(div_scalar_stub, binary_op_scalar)
HMP_DECLARE_DISPATCH_STUB(div_scalar_stub2, binary_op_scalar2)


Tensor& mul(Tensor &out, const Tensor &ina, const Tensor &inb);
Tensor& mul(Tensor &out, const Tensor &ina, const Scalar &inb);
Tensor& add(Tensor &out, const Tensor &ina, const Tensor &inb);
Tensor& add(Tensor &out, const Tensor &ina, const Scalar &inb);
Tensor& sub(Tensor &out, const Tensor &ina, const Tensor &inb);
Tensor& sub(Tensor &out, const Tensor &ina, const Scalar &inb);
Tensor& sub(Tensor &out, const Scalar &ina, const Tensor &inb);
Tensor& div(Tensor &out, const Tensor &ina, const Tensor &inb);
Tensor& div(Tensor &out, const Tensor &ina, const Scalar &inb);
Tensor& div(Tensor &out, const Scalar &ina, const Tensor &inb);


}} //namespace