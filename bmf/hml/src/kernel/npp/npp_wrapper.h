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

#include <npp.h>
#include <hmp/imgproc/image.h>

namespace hmp{
namespace kernel{

inline int getNPPIFilterMode(ImageFilterMode mode)
{
    switch(mode){
        case ImageFilterMode::Nearest:
            return NPPI_INTER_NN;
        case ImageFilterMode::Bilinear:
            return NPPI_INTER_LINEAR;
        case ImageFilterMode::Bicubic:
            return NPPI_INTER_CUBIC;
        default:
            HMP_REQUIRE(false, "unsupport filter mode({}) in NPPI", mode);
    }
}


inline double getNPPRotationAngle(ImageRotationMode rotate)
{
    switch(rotate){
        case ImageRotationMode::Rotate0:
            return 0;
        case ImageRotationMode::Rotate90:
            return 90;
        case ImageRotationMode::Rotate180:
            return 180;
        case ImageRotationMode::Rotate270:
            return 270;
        default:
            HMP_REQUIRE(false, "unsupport rotation mode({}) in NPP", rotate);
    }
}

inline NppiAxis getNPPAxis(ImageAxis axis)
{
    switch(axis){
        case ImageAxis::Horizontal:
            return NPP_VERTICAL_AXIS;
        case ImageAxis::Vertical:
            return NPP_HORIZONTAL_AXIS;
        case ImageAxis::HorizontalAndVertical:
            return NPP_BOTH_AXIS;
        default:
            HMP_REQUIRE(false, "unsupport axis({}) in NPP", axis);
    }
}


inline NppStreamContext makeNppStreamContext()
{
    auto stream = cuda::getCurrentCUDAStream();

    int deviceId;
    HMP_CUDA_CHECK(cudaGetDevice(&deviceId));

    cudaDeviceProp oDeviceProperties;
    HMP_CUDA_CHECK(cudaGetDeviceProperties(&oDeviceProperties, deviceId));

    unsigned int streamFlags;
    cudaStreamGetFlags(stream, &streamFlags);

    int computeCapabilityMajor;
    HMP_CUDA_CHECK(cudaDeviceGetAttribute(
      &computeCapabilityMajor, cudaDevAttrComputeCapabilityMajor, deviceId));

    int computeCapabilityMinor;
    HMP_CUDA_CHECK(cudaDeviceGetAttribute(
      &computeCapabilityMinor, cudaDevAttrComputeCapabilityMinor, deviceId));

    return NppStreamContext{
        .hStream = stream,
        .nCudaDeviceId = deviceId,
        .nMultiProcessorCount = oDeviceProperties.multiProcessorCount,
        .nMaxThreadsPerMultiProcessor = oDeviceProperties.maxThreadsPerMultiProcessor,
        .nMaxThreadsPerBlock = oDeviceProperties.maxThreadsPerBlock,
        .nSharedMemPerBlock = oDeviceProperties.sharedMemPerBlock,
        .nCudaDevAttrComputeCapabilityMajor = computeCapabilityMajor,
        .nCudaDevAttrComputeCapabilityMinor = computeCapabilityMinor,
        .nStreamFlags = streamFlags
    };
}


template<typename T>
void nppiResize(Tensor &dst, const Tensor &src, int mode, NppStreamContext &ctx)
{
    HMP_REQUIRE(false, "resize_cuda: unsupported scalar type {}", getScalarType<T>());
}

template<>
void nppiResize<uint8_t>(Tensor &dst, const Tensor &src, int mode, NppStreamContext &ctx)
{
    using scalar_t = uint8_t;

    NppiSize sSize{.width = (int)src.size(1), .height = (int)src.size(0)};
    NppiRect sRoi{.x = 0, .y = 0, .width = sSize.width, .height = sSize.height};
    NppiSize dSize{.width = (int)dst.size(1), .height = (int)dst.size(0)};
    NppiRect dRoi{.x = 0, .y = 0, .width = dSize.width, .height = dSize.height};
    auto status = nppiResize_8u_C1R_Ctx(
        src.data<scalar_t>(), src.stride(0) * sizeof(scalar_t), sSize, sRoi,
        dst.data<scalar_t>(), dst.stride(0) * sizeof(scalar_t), dSize, dRoi,
        mode, ctx);
    HMP_REQUIRE(status == NPP_SUCCESS, 
                "nppiResize_8u_C1R_Ctx failed with status={}", status);
}


template<>
void nppiResize<uint16_t>(Tensor &dst, const Tensor &src, int mode, NppStreamContext &ctx)
{
    using scalar_t = uint16_t;

    NppiSize sSize{.width = (int)src.size(1), .height = (int)src.size(0)};
    NppiRect sRoi{.x = 0, .y = 0, .width = sSize.width, .height = sSize.height};
    NppiSize dSize{.width = (int)dst.size(1), .height = (int)dst.size(0)};
    NppiRect dRoi{.x = 0, .y = 0, .width = dSize.width, .height = dSize.height};
    auto status = nppiResize_16u_C1R_Ctx(
        src.data<scalar_t>(), src.stride(0) * sizeof(scalar_t), sSize, sRoi,
        dst.data<scalar_t>(), dst.stride(0) * sizeof(scalar_t), dSize, dRoi,
        mode, ctx);
    HMP_REQUIRE(status == NPP_SUCCESS, 
                "nppiResize_16u_C1R_Ctx failed with status={}", status);
}


template<>
void nppiResize<float>(Tensor &dst, const Tensor &src, int mode, NppStreamContext &ctx)
{
    using scalar_t = float;

    NppiSize sSize{.width = (int)src.size(1), .height = (int)src.size(0)};
    NppiRect sRoi{.x = 0, .y = 0, .width = sSize.width, .height = sSize.height};
    NppiSize dSize{.width = (int)dst.size(1), .height = (int)dst.size(0)};
    NppiRect dRoi{.x = 0, .y = 0, .width = dSize.width, .height = dSize.height};
    auto status = nppiResize_32f_C1R_Ctx(
        src.data<scalar_t>(), src.stride(0) * sizeof(scalar_t), sSize, sRoi,
        dst.data<scalar_t>(), dst.stride(0) * sizeof(scalar_t), dSize, dRoi,
        mode, ctx);
    HMP_REQUIRE(status == NPP_SUCCESS, 
                "nppiResize_32f_C1R_Ctx failed with status={}", status);
}



template<typename T>
void nppiRotate(Tensor &dst, const Tensor &src, ImageRotationMode mode, NppStreamContext &ctx)
{
    HMP_REQUIRE(false, "rotate_cuda: unsupported scalar type {}", getScalarType<T>());
}

template<>
void nppiRotate<uint8_t>(Tensor &dst, const Tensor &src, ImageRotationMode mode, NppStreamContext &ctx)
{
    using scalar_t = uint8_t;

    auto angle = getNPPRotationAngle(mode);
    NppiSize sSize{.width = (int)src.size(1), .height = (int)src.size(0)};
    NppiRect sRoi{.x = 0, .y = 0, .width = sSize.width, .height = sSize.height};
    NppiSize dSize{.width = (int)dst.size(1), .height = (int)dst.size(0)};
    NppiRect dRoi{.x = 0, .y = 0, .width = dSize.width, .height = dSize.height};
    int xshift = (angle == 270 || angle == 180) ? dst.size(1) - 1 : 0;
    int yshift = (angle == 90 || angle == 180) ? dst.size(0) - 1 : 0;

    auto status = nppiRotate_8u_C1R_Ctx(src.data<scalar_t>(), sSize, src.stride(0) * sizeof(scalar_t), sRoi,
                                        dst.data<scalar_t>(), dst.stride(0) * sizeof(scalar_t), dRoi,
                                        angle, xshift, yshift, NPPI_INTER_NN, ctx);
    HMP_REQUIRE(status >= NPP_SUCCESS,
                "nppiRotate_8u_C1R_Ctx failed with status={}", status);
}


template<>
void nppiRotate<uint16_t>(Tensor &dst, const Tensor &src, ImageRotationMode mode, NppStreamContext &ctx)
{
    using scalar_t = uint16_t;

    auto angle = getNPPRotationAngle(mode);
    NppiSize sSize{.width = (int)src.size(1), .height = (int)src.size(0)};
    NppiRect sRoi{.x = 0, .y = 0, .width = sSize.width, .height = sSize.height};
    NppiSize dSize{.width = (int)dst.size(1), .height = (int)dst.size(0)};
    NppiRect dRoi{.x = 0, .y = 0, .width = dSize.width, .height = dSize.height};
    int xshift = (angle == 270 || angle == 180) ? dst.size(1) - 1 : 0;
    int yshift = (angle == 90 || angle == 180) ? dst.size(0) - 1 : 0;

    auto status = nppiRotate_16u_C1R_Ctx(src.data<scalar_t>(), sSize, src.stride(0) * sizeof(scalar_t), sRoi,
                                        dst.data<scalar_t>(), dst.stride(0) * sizeof(scalar_t), dRoi,
                                        angle, xshift, yshift, NPPI_INTER_NN, ctx);
    HMP_REQUIRE(status >= NPP_SUCCESS,
                "nppiRotate_16u_C1R_Ctx failed with status={}", status);
}


template<>
void nppiRotate<float>(Tensor &dst, const Tensor &src, ImageRotationMode mode, NppStreamContext &ctx)
{
    using scalar_t = float;

    auto angle = getNPPRotationAngle(mode);
    NppiSize sSize{.width = (int)src.size(1), .height = (int)src.size(0)};
    NppiRect sRoi{.x = 0, .y = 0, .width = sSize.width, .height = sSize.height};
    NppiSize dSize{.width = (int)dst.size(1), .height = (int)dst.size(0)};
    NppiRect dRoi{.x = 0, .y = 0, .width = dSize.width, .height = dSize.height};
    int xshift = (angle == 270 || angle == 180) ? dst.size(1) - 1 : 0;
    int yshift = (angle == 90 || angle == 180) ? dst.size(0) - 1 : 0;

    auto status = nppiRotate_32f_C1R_Ctx(src.data<scalar_t>(), sSize, src.stride(0) * sizeof(scalar_t), sRoi,
                                        dst.data<scalar_t>(), dst.stride(0) * sizeof(scalar_t), dRoi,
                                        angle, xshift, yshift, NPPI_INTER_NN, ctx);
    HMP_REQUIRE(status >= NPP_SUCCESS,
                "nppiRotate_32f_C1R_Ctx failed with status={}", status);
}


template<typename T>
void nppiMirror(Tensor &dst, const Tensor &src, ImageAxis, NppStreamContext &ctx)
{
    HMP_REQUIRE(false, "mirror_cuda: unsupported scalar type {}", getScalarType<T>());
}

template<>
void nppiMirror<uint8_t>(Tensor &dst, const Tensor &src, ImageAxis axis, NppStreamContext &ctx)
{
    using scalar_t = uint8_t;

    auto flip = getNPPAxis(axis);
    NppiSize oROI{.width = (int)dst.size(1), .height = (int)dst.size(0)};
    auto status = nppiMirror_8u_C1R_Ctx(src.data<scalar_t>(), src.stride(0) * sizeof(scalar_t),
                                        dst.data<scalar_t>(), dst.stride(0) * sizeof(scalar_t), oROI, 
                                        flip, ctx);
    HMP_REQUIRE(status >= NPP_SUCCESS,
                "nppiMirror_8u_C1R_Ctx failed with status={}", status);
}

template<>
void nppiMirror<uint16_t>(Tensor &dst, const Tensor &src, ImageAxis axis, NppStreamContext &ctx)
{
    using scalar_t = uint16_t;

    auto flip = getNPPAxis(axis);
    NppiSize oROI{.width = (int)dst.size(1), .height = (int)dst.size(0)};
    auto status = nppiMirror_16u_C1R_Ctx(src.data<scalar_t>(), src.stride(0) * sizeof(scalar_t),
                                        dst.data<scalar_t>(), dst.stride(0) * sizeof(scalar_t), oROI, 
                                        flip, ctx);
    HMP_REQUIRE(status >= NPP_SUCCESS,
                "nppiMirror_16u_C1R_Ctx failed with status={}", status);
}


template<>
void nppiMirror<float>(Tensor &dst, const Tensor &src, ImageAxis axis, NppStreamContext &ctx)
{
    using scalar_t = float;

    auto flip = getNPPAxis(axis);
    NppiSize oROI{.width = (int)dst.size(1), .height = (int)dst.size(0)};
    auto status = nppiMirror_32f_C1R_Ctx(src.data<scalar_t>(), src.stride(0) * sizeof(scalar_t),
                                        dst.data<scalar_t>(), dst.stride(0) * sizeof(scalar_t), oROI, 
                                        flip, ctx);
    HMP_REQUIRE(status >= NPP_SUCCESS,
                "nppiMirror_32f_C1R_Ctx failed with status={}", status);
}


}} //namespace