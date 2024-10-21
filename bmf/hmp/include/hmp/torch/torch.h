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
#pragma once
#include <hmp/tensor.h>
#include <torch/torch.h>

namespace hmp {
namespace torch {

HMP_API at::Tensor tensor(const Tensor &from);
HMP_API Tensor from_tensor(const at::Tensor &from);
} // namespace torch
} // namespace hmp
