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

#include <pybind11/pybind11.h>
#include <hmp/tensor.h>

hmp::TensorOptions parse_tensor_options(const pybind11::kwargs &kwargs, const hmp::TensorOptions &ref = {});

hmp::Device parse_device(const pybind11::object &obj, const hmp::Device &device = hmp::kCPU);