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
#include <hmp/torch/torch.h>


namespace hmp{
namespace torch{

static inline c10::ScalarType typeMetaToScalarType(const caffe2::TypeMeta& dtype)
{
#if TORCH_VERSION_MAJOR >= 1 && TORCH_VERSION_MINOR >= 8
    return dtype.toScalarType();
#else
    return c10::typeMetaToScalarType(dtype);
#endif
}

inline c10::ScalarType scalar_type(const ScalarType s)
{
    switch(s){
        case kUInt8: return c10::kByte;
        case kInt8: return c10::kChar;
        case kInt16: return c10::kShort;
        case kInt32: return c10::kInt;
        case kInt64: return c10::kLong;
        case kHalf: return c10::kHalf;
        case kFloat32: return c10::kFloat;
        case kFloat64: return c10::kDouble;
        default:
            HMP_REQUIRE(false, "hmp scalar type {} is not supported by torch", s);
    }
}

inline ScalarType from_scalar_type(const c10::ScalarType &s)
{
    switch(s){
        case c10::kByte: return kUInt8;
        case c10::kChar: return kInt8;
        case c10::kShort: return kInt16;
        case c10::kInt: return kInt32;
        case c10::kLong: return kInt64;
        case c10::kHalf: return kHalf;
        case c10::kFloat: return kFloat32;
        case c10::kDouble: return kFloat64;
        default:
            HMP_REQUIRE(false, "torch scalar type {} is not supported by hmp", s);
    }
}


inline ScalarType from_scalar_type(const caffe2::TypeMeta &s)
{
    return from_scalar_type(typeMetaToScalarType(s));
}

inline c10::DeviceType device_type(DeviceType d)
{
    switch(d){
        case kCPU: return c10::kCPU;
        case kCUDA: return c10::kCUDA;
        default:
            HMP_REQUIRE(false, "hmp device type {} is not supported by torch", d);
    }
}


inline DeviceType from_device_type(c10::DeviceType d)
{
    switch(d){
        case c10::kCPU: return kCPU;
        case c10::kCUDA: return kCUDA;
        default:
            HMP_REQUIRE(false, "torch device type {} is not supported by hmp", d);
    }
}


inline c10::Device device(const Device &d)
{
    auto index = d.type() == kCPU ? -1 : d.index();
    return c10::Device(device_type(d.type()), index);
}

inline Device from_device(const c10::Device &d)
{
    auto index = d.index() < 0 ? 0 : d.index();
    return Device(from_device_type(d.type()), index);
}


inline c10::TensorOptions tensor_options(const TensorOptions &options)
{
    return c10::TensorOptions()
            .device(device(options.device()))
            .dtype(scalar_type(options.scalar_type()))
            .pinned_memory(options.pinned_memory());
}


inline TensorOptions from_tensor_options(const c10::TensorOptions &options)
{
    return TensorOptions()
            .device(from_device(options.device()))
            .scalar_type(from_scalar_type(options.dtype()))
            .pinned_memory(options.pinned_memory());
}


at::Tensor tensor(const Tensor &from)
{
    auto info = from.tensorInfo();

    return at::from_blob(
        info->unsafe_data(),
        info->shape(),
        info->strides(),
        [info](void *){},
        tensor_options(from.options())
    );
}


Tensor from_tensor(const at::Tensor &from)
{
    auto sizes = from.sizes();
    auto strides = from.strides();   

    return from_buffer(
        DataPtr(from.data_ptr(), [from](void*){}, from_device(from.device())),
        from_scalar_type(from.dtype()),
        SizeArray(sizes.begin(), sizes.end()),
        SizeArray(strides.begin(), strides.end())
    );
}



}} //


