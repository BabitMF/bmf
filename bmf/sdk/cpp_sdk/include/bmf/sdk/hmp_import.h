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
