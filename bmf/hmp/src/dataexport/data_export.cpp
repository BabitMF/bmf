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
#include <hmp/dataexport/data_export.h>

#include <iostream>
#include <string>

namespace hmp {
static DLDataType get_dl_dtype(const Tensor &t) {
    DLDataType dtype;
    dtype.lanes = 1;
    dtype.bits = sizeof_scalar_type(t.scalar_type()) * 8;
    switch (t.scalar_type()) {
    case ScalarType::UInt8:
    case ScalarType::UInt16:
        dtype.code = DLDataTypeCode::kDLUInt;
        break;
    case ScalarType::Int8:
    case ScalarType::Int16:
    case ScalarType::Int32:
    case ScalarType::Int64:
        dtype.code = DLDataTypeCode::kDLInt;
        break;
    case ScalarType::Float64:
    case ScalarType::Float32:
    case ScalarType::Half:
        dtype.code = DLDataTypeCode::kDLFloat;
        break;
    case ScalarType::Undefined:
    default:
        HMP_REQUIRE(false, "Undefined is not a valid ScalarType");
    }
    return dtype;
}

ScalarType to_scalar_type(const DLDataType &dtype) {
    ScalarType stype;
    HMP_REQUIRE(dtype.lanes == 1, "hmp does not support lanes != 1");
    switch (dtype.code) {
    case DLDataTypeCode::kDLUInt:
        switch (dtype.bits) {
        case 8:
            stype = ScalarType::UInt8;
            break;
        case 16:
            stype = ScalarType::UInt16;
            break;
        default:
            HMP_REQUIRE(false,
                        "Unsupported kUInt bits " + std::to_string(dtype.bits));
        }
        break;
    case DLDataTypeCode::kDLInt:
        switch (dtype.bits) {
        case 8:
            stype = ScalarType::Int8;
            break;
        case 16:
            stype = ScalarType::Int16;
            break;
        case 32:
            stype = ScalarType::Int32;
            break;
        case 64:
            stype = ScalarType::Int64;
            break;
        default:
            HMP_REQUIRE(false,
                        "Unsupported kInt bits " + std::to_string(dtype.bits));
        }
        break;
    case DLDataTypeCode::kDLFloat:
        switch (dtype.bits) {
        case 16:
            stype = ScalarType::Half;
            break;
        case 32:
            stype = ScalarType::Float32;
            break;
        case 64:
            stype = ScalarType::Float64;
            break;
        default:
            HMP_REQUIRE(false, "Unsupported kFloat bits " +
                                   std::to_string(dtype.bits));
        }
        break;
    default:
        HMP_REQUIRE(false, "Unsupported code " + std::to_string(dtype.code));
    }
    return stype;
}

static DLDevice get_dl_device(const Tensor &tensor, const int64_t &device_id) {
    DLDevice ctx;
    ctx.device_id = device_id;
    switch (tensor.device().type()) {
    case DeviceType::CPU:
        ctx.device_type = DLDeviceType::kDLCPU;
        break;
    case DeviceType::CUDA:
        ctx.device_type = DLDeviceType::kDLCUDA;
        break;
    default:
        HMP_REQUIRE(false,
                    "Cannot pack tensors on " + stringfy(tensor.device()));
    }
    return ctx;
}

static Device get_hmp_device(const DLDevice &ctx) {
    switch (ctx.device_type) {
    case DLDeviceType::kDLCPU:
        return Device(DeviceType::CPU);
#ifdef HMP_ENABLE_CUDA
    case DLDeviceType::kDLCUDA:
        return Device(DeviceType::CUDA, ctx.device_id);
#endif
    default:
        HMP_REQUIRE(false, "Unsupported device_type: " +
                               std::to_string(ctx.device_type));
    }
}

struct HmpDLMTensor {
    Tensor handle;
    DLManagedTensor tensor;
};

void deleter(DLManagedTensor *arg) {
    delete static_cast<HmpDLMTensor *>(arg->manager_ctx);
}

DLManagedTensor *to_dlpack(const Tensor &src) {

    HmpDLMTensor *hmpDLMTensor(new HmpDLMTensor);
    hmpDLMTensor->handle = src;
    hmpDLMTensor->tensor.manager_ctx = hmpDLMTensor;
    hmpDLMTensor->tensor.deleter = &deleter;
    hmpDLMTensor->tensor.dl_tensor.data = src.unsafe_data();
    int64_t device_id = 0;
    if (src.is_cuda()) {
        device_id = src.device_index();
    }
    hmpDLMTensor->tensor.dl_tensor.device = get_dl_device(src, device_id);
    hmpDLMTensor->tensor.dl_tensor.ndim = src.dim();
    hmpDLMTensor->tensor.dl_tensor.dtype = get_dl_dtype(src);
    hmpDLMTensor->tensor.dl_tensor.shape =
        const_cast<int64_t *>(src.shape().data());
    hmpDLMTensor->tensor.dl_tensor.strides =
        const_cast<int64_t *>(src.strides().data());
    hmpDLMTensor->tensor.dl_tensor.byte_offset = 0;
    return &(hmpDLMTensor->tensor);
}

Tensor from_dlpack(const DLManagedTensor *src) {
    Device device = get_hmp_device(src->dl_tensor.device);
    ScalarType stype = to_scalar_type(src->dl_tensor.dtype);
    DataPtr dp{src->dl_tensor.data, device};
    SizeArray shape{src->dl_tensor.shape,
                    src->dl_tensor.shape + src->dl_tensor.ndim};
    if (!src->dl_tensor.strides) {
        return from_buffer({src->dl_tensor.data, device}, stype, shape);
    }
    SizeArray strides{src->dl_tensor.strides,
                      src->dl_tensor.strides + src->dl_tensor.ndim};
    return from_buffer({src->dl_tensor.data, device}, stype, shape, strides);
}
} // namespace hmp