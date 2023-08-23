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
#include <hmp/imgproc.h>
#include <hmp/format.h>
#ifdef HMP_ENABLE_CUDA
#include <hmp/cuda/allocator.h>
#endif
#include <bmf/sdk/bmf_type_info.h>

namespace bmf_sdk {

using hmp::ScalarType;
// helper enums types
#define DEFINE_KSCALAR(_, S) using hmp::k##S;
HMP_FORALL_SCALAR_TYPES(DEFINE_KSCALAR)
#undef DEFINE_KSCALAR

using hmp::Device;
using hmp::DeviceGuard;
using hmp::kCPU;
using hmp::kCUDA;

using hmp::DataPtr;
using hmp::DeviceType;
using hmp::SizeArray;
using hmp::Tensor;
using hmp::TensorList;
using hmp::TensorOptions;

using hmp::makeRefPtr;
using hmp::RefObject;
using hmp::RefPtr;

using hmp::ChannelFormat;
using hmp::ColorPrimaries;
using hmp::ColorRange;
using hmp::ColorSpace;
using hmp::ColorTransferCharacteristic;
using hmp::PixelFormat;
using hmp::PixelInfo;

using hmp::Frame;

using hmp::kNCHW;
using hmp::kNHWC;

#ifdef HMP_ENABLE_CUDA
using namespace hmp::cuda;
#endif

} // namespace bmf_sdk

BMF_DEFINE_TYPE(hmp::Tensor)
