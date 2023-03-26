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


Tensor& concat(Tensor &out, const TensorList &tensors, int64_t axis = 0);
Tensor concat(const TensorList &tensors, int64_t axis = 0);

Tensor& stack(Tensor &out, const TensorList &tensors, int64_t axis = 0);
Tensor stack(const TensorList &tensors, int64_t axis = 0);

Tensor& vstack(Tensor &out, const TensorList &tensors);
Tensor vstack(const TensorList &tensors);

Tensor& hstack(Tensor &out, const TensorList &tensors);
Tensor hstack(const TensorList &tensors);

}} //namespace