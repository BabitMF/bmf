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

#include <libyuv.h> 
#include <hmp/imgproc/image.h>

namespace hmp{
namespace kernel{

static libyuv::FilterModeEnum getLibYUVFilterMode(ImageFilterMode mode)
{
    switch(mode){
        case ImageFilterMode::Nearest:
            return libyuv::kFilterNone;
        case ImageFilterMode::Bilinear:
            return libyuv::kFilterBilinear;
        case ImageFilterMode::Bicubic:
            return libyuv::kFilterBox;
        default:
            HMP_REQUIRE(false, "unsupport filter mode({}) in libyuv", mode);
    }
}

static libyuv::RotationMode getLibYUVRotationMode(ImageRotationMode rotate)
{
    switch(rotate){
        case ImageRotationMode::Rotate0:
            return libyuv::kRotate0;
        case ImageRotationMode::Rotate90:
            return libyuv::kRotate90;
        case ImageRotationMode::Rotate180:
            return libyuv::kRotate180;
        case ImageRotationMode::Rotate270:
            return libyuv::kRotate270;
        default:
            HMP_REQUIRE(false, "unsupport rotation mode({}) in libyuv", rotate);
    }
}


template<typename T>
inline void libyuvScalePlane(Tensor &dst, const Tensor &src, libyuv::FilterModeEnum mode)
{
    HMP_REQUIRE(false, "im_scale_cpu: scale type {} is not supported", getScalarType<T>());
}

template<>
inline void libyuvScalePlane<uint8_t>(Tensor &dst, const Tensor &src, libyuv::FilterModeEnum mode)
{
    using scalar_t = uint8_t;
    libyuv::ScalePlane(src.data<scalar_t>(), src.stride(0), src.size(1), src.size(0), 
                       dst.data<scalar_t>(), dst.stride(0), dst.size(1), dst.size(0),
                       mode);
}

template<>
inline void libyuvScalePlane<uint16_t>(Tensor &dst, const Tensor &src, libyuv::FilterModeEnum mode)
{
    using scalar_t = uint16_t;
    libyuv::ScalePlane_16(src.data<scalar_t>(), src.stride(0) * sizeof(scalar_t), src.size(1), src.size(0), 
                       dst.data<scalar_t>(), dst.stride(0) * sizeof(scalar_t), dst.size(1), dst.size(0),
                       mode);
}


template<typename T>
inline void libyuvRotatePlane(Tensor &dst, const Tensor &src, libyuv::RotationModeEnum mode)
{
    HMP_REQUIRE(false, "im_rotate_cpu: scale type {} is not supported", getScalarType<T>());
}

template<>
inline void libyuvRotatePlane<uint8_t>(Tensor &dst, const Tensor &src, libyuv::RotationModeEnum mode)
{
    using scalar_t = uint8_t;
    libyuv::RotatePlane(src.data<scalar_t>(), src.stride(0) * sizeof(scalar_t),
                       dst.data<scalar_t>(), dst.stride(0) * sizeof(scalar_t), src.size(1), src.size(0), mode);
}


template<typename T>
inline void libyuvMirrorPlane(Tensor &dst, const Tensor &src, ImageAxis axis)
{
    HMP_REQUIRE(false, "im_mirror_cpu: scale type {} is not supported", getScalarType<T>());
}

template<>
inline void libyuvMirrorPlane<uint8_t>(Tensor &dst, const Tensor &src, ImageAxis axis)
{
    using scalar_t = uint8_t;
    if(axis == ImageAxis::Horizontal || axis == ImageAxis::HorizontalAndVertical){
        auto height = axis==ImageAxis::HorizontalAndVertical ? -src.size(0) : src.size(0);
        libyuv::MirrorPlane(src.data<scalar_t>(), src.stride(0) * sizeof(scalar_t),
                           dst.data<scalar_t>(), dst.stride(0) * sizeof(scalar_t), src.size(1), height);
    }
    else{
        libyuv::RotatePlane(src.data<scalar_t>(), src.stride(0) * sizeof(scalar_t),
                       dst.data<scalar_t>(), dst.stride(0) * sizeof(scalar_t),
                       src.size(1), -src.size(0), libyuv::kRotate0);
    }
}





}} //namespace hmp::kernel