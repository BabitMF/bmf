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

#include <hmp/imgproc/image.h>
#include <kernel/dispatch_stub.h>

namespace hmp{

//https://chromium.googlesource.com/libyuv/libyuv/+/master/docs/formats.md
// ie. I420, //YUV420, 8bit, BT.601 limited range
//     H420, //YUV420, 8bit, BT.709 limited range


#define HMP_FORALL_PPIXEL_FORMATS(_)  \
    _(I420)                         \
    _(I422)                         \
    _(I444)                         \
    _(H420)                         \
    _(H422)                         \
    _(H444)                         \
    _(NV12)                         \
    _(NV21)                         \
    _(RGBA)                         \
    _(NV12_BT709)                   \
    _(NV21_BT709)

//internal use only
enum class PPixelFormat : uint8_t{
    //update sPixelFormatMetas if you modify this structure
#define DEFINE_ENUM(name) name,
    HMP_FORALL_PPIXEL_FORMATS(DEFINE_ENUM)
#undef DEFINE_ENUM
};


std::string stringfy(const PPixelFormat &format);


namespace kernel{

HMP_DECLARE_DISPATCH_STUB(yuv_to_rgb_stub, 
        Tensor&(*)(Tensor&, const TensorList&, PPixelFormat, ChannelFormat));
HMP_DECLARE_DISPATCH_STUB(rgb_to_yuv_stub, 
        TensorList&(*)(TensorList&, const Tensor&, PPixelFormat, ChannelFormat));
HMP_DECLARE_DISPATCH_STUB(yuv_resize_stub, 
        TensorList&(*)(TensorList&, const TensorList&, PPixelFormat, ImageFilterMode));
HMP_DECLARE_DISPATCH_STUB(yuv_rotate_stub, 
        TensorList&(*)(TensorList&, const TensorList&, PPixelFormat, ImageRotationMode));
HMP_DECLARE_DISPATCH_STUB(yuv_mirror_stub,
        TensorList&(*)(TensorList&, const TensorList&, PPixelFormat, ImageAxis));
HMP_DECLARE_DISPATCH_STUB(img_resize_stub, 
        Tensor&(*)(Tensor&, const Tensor&, ImageFilterMode, ChannelFormat));
HMP_DECLARE_DISPATCH_STUB(img_rotate_stub, 
        Tensor&(*)(Tensor&, const Tensor&, ImageRotationMode, ChannelFormat));
HMP_DECLARE_DISPATCH_STUB(img_mirror_stub,
        Tensor&(*)(Tensor&, const Tensor&, ImageAxis, ChannelFormat));
HMP_DECLARE_DISPATCH_STUB(img_normalize_stub, 
        Tensor&(*)(Tensor&, const Tensor&, const Tensor&, const Tensor&, ChannelFormat));
HMP_DECLARE_DISPATCH_STUB(img_erode_stub,
        Tensor&(*)(Tensor&, const Tensor&, const Tensor&, ChannelFormat));
HMP_DECLARE_DISPATCH_STUB(img_dilate_stub,
        Tensor&(*)(Tensor&, const Tensor&, const Tensor&, ChannelFormat));
HMP_DECLARE_DISPATCH_STUB(img_sobel_stub,
        Tensor&(*)(Tensor&, const Tensor&, int64_t, int64_t, int64_t, 
        const Scalar&, const Scalar&, ChannelFormat));
HMP_DECLARE_DISPATCH_STUB(img_canny_stub,
        Tensor&(*)(Tensor&, const Tensor&, const Scalar&, const Scalar&, 
        int64_t, bool, ChannelFormat));
HMP_DECLARE_DISPATCH_STUB(img_filter2d_stub, 
        Tensor&(*)(Tensor&, const Tensor&, const Tensor&, const Scalar&, ChannelFormat));
HMP_DECLARE_DISPATCH_STUB(img_gaussian_blur_stub, 
        Tensor&(*)(Tensor&, const Tensor&, int, int, const Scalar&, const Scalar&, ChannelFormat));
HMP_DECLARE_DISPATCH_STUB(img_bilateral_filter_stub, 
        Tensor&(*)(Tensor&, const Tensor&, int, const Scalar&, const Scalar&, ChannelFormat));

HMP_DECLARE_DISPATCH_STUB(img_warp_perspective_stub, 
        Tensor&(*)(Tensor&, const Tensor&, const Tensor&, ImageFilterMode, ChannelFormat));


Tensor &yuv_to_rgb(Tensor &dst, const TensorList &src, PPixelFormat pformat, ChannelFormat cformat);
TensorList &rgb_to_yuv(TensorList &dst, const Tensor &src, PPixelFormat pformat, ChannelFormat cformat);

TensorList &yuv_resize(TensorList &dst, const TensorList &src,
                          PPixelFormat format, ImageFilterMode mode = ImageFilterMode::Bilinear);
TensorList &yuv_rotate(TensorList &dst, const TensorList &src,
                           PPixelFormat format, ImageRotationMode rotate);
TensorList &yuv_mirror(TensorList &dst, const TensorList &src, PPixelFormat format, ImageAxis axis);


Tensor &img_resize(Tensor &dst, const Tensor &src, ImageFilterMode mode, ChannelFormat cformat);
Tensor &img_rotate(Tensor &dst, const Tensor &src, ImageRotationMode mode, ChannelFormat cformat);
Tensor &img_mirror(Tensor &dst, const Tensor &src, ImageAxis axis, ChannelFormat cformat);

Tensor& img_normalize(Tensor &dst, const Tensor &src, const Tensor &mean, const Tensor &std, ChannelFormat cformat=kNCHW);

Tensor &img_erode(Tensor &dst, const Tensor &src, const Tensor &kernel,
    ChannelFormat cformat=kNCHW);
Tensor &img_dilate(Tensor &dst, const Tensor &src, const Tensor &kernel,
    ChannelFormat cformat=kNCHW);

Tensor &img_sobel(Tensor &dst, const Tensor &src, int64_t dx, int64_t dy,
    int64_t ksize = 3, const Scalar& scale = 1, const Scalar& delta=0,
    ChannelFormat cformat=kNCHW);

Tensor &img_canny(Tensor &dst, const Tensor &src, 
    const Scalar &low_thresh, const Scalar &high_thresh,
    int64_t aperture_size = 3, bool l2_gradient = false,
    ChannelFormat cformat = kNCHW); 


Tensor &img_filter2d(Tensor &dst, const Tensor &src,
    const Tensor &kernel, const Scalar& delta=0, ChannelFormat cformat=kNCHW);

Tensor &img_overlay(Tensor &dst, const Tensor &src0, const Tensor &src1, 
    const Tensor &alpha, ChannelFormat cformat=kNCHW);

Tensor& img_bilateral_filter(Tensor &dst, const Tensor &src, int d, 
    const Scalar& sigma_color, const Scalar& sigma_space, ChannelFormat cformat=kNCHW);

Tensor& img_gaussian_blur(Tensor &dst, const Tensor &src, int kx, int ky,
     const Scalar& sigma_x, const Scalar& sigma_y=0, ChannelFormat cformat=kNCHW);

Tensor &img_warp_perspective(Tensor &dst, const Tensor &src,
    const Tensor &M, ImageFilterMode mode = kBicubic, ChannelFormat cformat=kNCHW);


}} //
