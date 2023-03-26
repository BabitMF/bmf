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

#include <hmp/imgproc.h>
#include <kernel/kernel_utils.h>
#include <kernel/imgproc.h>

namespace hmp{


std::string stringfy(const PPixelFormat &format)
{
    switch (format)
    {
#define STRINGFY_CASE(name) case PPixelFormat::name : return "k"#name;
    HMP_FORALL_PPIXEL_FORMATS(STRINGFY_CASE)
#undef STRINGFY_CASE
    default:
        return fmt::format("PPixelFormat({})", static_cast<int>(format));
    }
}


namespace kernel{

HMP_DEFINE_DISPATCH_STUB(yuv_to_rgb_stub)
HMP_DEFINE_DISPATCH_STUB(rgb_to_yuv_stub)
HMP_DEFINE_DISPATCH_STUB(yuv_resize_stub)
HMP_DEFINE_DISPATCH_STUB(yuv_rotate_stub)
HMP_DEFINE_DISPATCH_STUB(yuv_mirror_stub)
HMP_DEFINE_DISPATCH_STUB(img_resize_stub)
HMP_DEFINE_DISPATCH_STUB(img_rotate_stub)
HMP_DEFINE_DISPATCH_STUB(img_mirror_stub)
HMP_DEFINE_DISPATCH_STUB(img_normalize_stub)
HMP_DEFINE_DISPATCH_STUB(img_erode_stub)
HMP_DEFINE_DISPATCH_STUB(img_dilate_stub)
HMP_DEFINE_DISPATCH_STUB(img_sobel_stub)
HMP_DEFINE_DISPATCH_STUB(img_canny_stub)
HMP_DEFINE_DISPATCH_STUB(img_filter2d_stub)
HMP_DEFINE_DISPATCH_STUB(img_gaussian_blur_stub)
HMP_DEFINE_DISPATCH_STUB(img_bilateral_filter_stub)
HMP_DEFINE_DISPATCH_STUB(img_warp_perspective_stub)

namespace {



static inline void img_common_check(const Tensor &tensor, ChannelFormat cformat,
     int64_t idx, const std::string &name)
{
    if(cformat == ChannelFormat::NHWC){
        HMP_REQUIRE(tensor.stride(-1) == 1,
                 "{}: expect {}th image tensor's channel stride is contiguous(0), got {}",
                 name, idx, tensor.stride(-1));
        HMP_REQUIRE(tensor.stride(-2) == tensor.size(-1),
                 "{}: expect {}th image tensor's width stride is contiguous({}), got {}",
                  name, idx, tensor.size(-1), tensor.stride(-2));
    }
    else{
        HMP_REQUIRE(tensor.stride(-1) == 1,
                 "{}: expect {}th image tensor's width stride is contiguous(1), got {}",
                  name, idx, tensor.stride(-1));
    }
}

static inline void img_common_check(const Tensor &dst, const Tensor &src, 
        ChannelFormat cformat, const std::string &name)
{
    checkDevice({src, dst}, dst.device(), name);
    img_common_check(dst, cformat, 0, name);
    img_common_check(src, cformat, 1, name);

    HMP_REQUIRE(src.size(0) == dst.size(0),
                "{}: expect src and dst image have same batch dim, got src={}, dst={}",
                name, src.size(0), dst.size(0));
}

static inline void yuv_common_check(const TensorList &tensors, size_t idx, const std::string &name)
{
    for(size_t i = 0; i < tensors.size(); ++i){
        img_common_check(tensors[i], ChannelFormat::NHWC, idx, name);
    }
}

static inline void yuv_common_check(const TensorList &dst, const TensorList &src, const std::string &name)
{
    HMP_REQUIRE(dst.size() == src.size(), 
        "{}: expect src and dst have same planes, got src={}, dst={}", name, src.size(), dst.size());
    for(size_t i = 0; i < dst.size(); ++i){
        img_common_check(dst[i], src[i], ChannelFormat::NHWC, name);
    }
}

} //namespace

Tensor &yuv_to_rgb(Tensor &dst, const TensorList &src, PPixelFormat pformat, ChannelFormat cformat)
{
    auto stmp = img::image_format(src, kNHWC);
    auto dtmp = img::image_format(dst, cformat);

    yuv_common_check(stmp, 1, "yuv_to_rgb");
    img_common_check(dtmp, cformat, 0, "yuv_to_rgb");
    auto cdim = cformat == kNCHW ? -3 : -1;
    HMP_REQUIRE(dtmp.size(cdim) == 3, "yuv_to_rgb: require 3 channels for dtmp, got {}", dtmp.size(cdim));

    //
    yuv_to_rgb_stub(dtmp.device_type(), dtmp, stmp, pformat, cformat);
    //

    return dst;
}

TensorList &rgb_to_yuv(TensorList &dst, const Tensor &src, PPixelFormat pformat, ChannelFormat cformat)
{
    auto stmp = img::image_format(src, cformat);
    auto dtmp = img::image_format(dst, kNHWC);

    yuv_common_check(dtmp, 0, "rgb_to_yuv");
    img_common_check(stmp, cformat, 1, "rgb_to_yuv");
    auto cdim = cformat == kNCHW ? -3 : -1;
    HMP_REQUIRE(stmp.size(cdim) == 3, "rgb_to_yuv: require 3 channels for dst, got {}", stmp.size(cdim));

    rgb_to_yuv_stub(stmp.device_type(), dtmp, stmp, pformat, cformat);

    return dst;
}

TensorList &yuv_resize(TensorList &dst, const TensorList &src,
                          PPixelFormat format, ImageFilterMode mode)
{
    auto stmp = img::image_format(src, kNHWC);
    auto dtmp = img::image_format(dst, kNHWC);

    yuv_common_check(dtmp, stmp, "yuv_resize");

    yuv_resize_stub(dtmp[0].device_type(), dtmp, stmp, format, mode);

    return dst;
}


TensorList &yuv_rotate(TensorList &dst, const TensorList &src, PPixelFormat format, ImageRotationMode rotate)
{
    auto stmp = img::image_format(src, kNHWC);
    auto dtmp = img::image_format(dst, kNHWC);

    yuv_common_check(dtmp, stmp, "yuv_rotate");

    if(rotate == ImageRotationMode::Rotate0 || rotate == ImageRotationMode::Rotate180){
        HMP_REQUIRE(dtmp[0].size(1) == stmp[0].size(1) && dtmp[0].size(2) == stmp[0].size(2),
             "yuv_rotate: image size are not matched with rotatation mode");
    }
    else if(rotate == ImageRotationMode::Rotate90 || rotate == ImageRotationMode::Rotate270){
        HMP_REQUIRE(dtmp[0].size(1) == stmp[0].size(2) && dtmp[0].size(2) == stmp[0].size(1),
             "yuv_rotate: image size are not matched with rotatation mode");
    }
    else{
        HMP_REQUIRE(false, "yuv_rotate: internal error")
    }

    yuv_rotate_stub(dtmp[0].device_type(), dtmp, stmp, format, rotate);

    return dst;
}


TensorList &yuv_mirror(TensorList &dst, const TensorList &src, PPixelFormat format, ImageAxis axis)
{
    auto stmp = img::image_format(src, kNHWC);
    auto dtmp = img::image_format(dst, kNHWC);

    yuv_common_check(dtmp, stmp, "yuv_mirror");
    HMP_REQUIRE(stmp[0].shape() == dtmp[0].shape(),
        "yuv_mirror: expect src and dst have same shape, got src={}, dst={}",
         stmp[0].shape(), dtmp[0].shape());

    yuv_mirror_stub(stmp[0].device_type(), dtmp, stmp, format, axis);

    return dst;
}


Tensor &img_resize(Tensor &dst, const Tensor &src, ImageFilterMode mode, ChannelFormat cformat)
{
    auto stmp = img::image_format(src, cformat);
    auto dtmp = img::image_format(dst, cformat);

    img_common_check(dtmp, stmp, cformat, "im_resize");
    HMP_REQUIRE(stmp.size(0) == dtmp.size(0), 
        "image_resize: expect same size of batch dim, src={}, dst={}", stmp.shape(), dtmp.shape());
    auto cdim = cformat == kNCHW ? 1 : -1;
    HMP_REQUIRE(stmp.size(cdim) == dtmp.size(cdim), 
        "image_resize: expect same size of channel dim, src={}, dst={}", stmp.shape(), dtmp.shape());

    img_resize_stub(dtmp.device_type(), dtmp, stmp, mode, cformat);

    return dst;
}

Tensor &img_rotate(Tensor &dst, const Tensor &src, ImageRotationMode mode, ChannelFormat cformat)
{
    auto stmp = img::image_format(src, cformat);
    auto dtmp = img::image_format(dst, cformat);

    img_common_check(dtmp, stmp, cformat, "im_rotate");

    int64_t cdim, wdim, hdim;
    if(cformat == kNCHW){
        wdim = -1;
        hdim = -2;
        cdim = -3;
    }
    else{
        cdim = -1;
        wdim = -2;
        hdim = -3;
    }
    HMP_REQUIRE(stmp.size(cdim) == dtmp.size(cdim), 
        "image_rotate: expect same size of channel dim, src={}, dst={}", stmp.shape(), dtmp.shape());

    bool shapeChanged = mode == ImageRotationMode::Rotate90 || mode == ImageRotationMode::Rotate270;
    if(shapeChanged){
        HMP_REQUIRE(stmp.size(hdim) == dtmp.size(wdim) && stmp.size(wdim) == dtmp.size(hdim), 
            "img_rotate: image size is invalid, expec {}, got {}", 
            SizeArray{stmp.size(wdim), stmp.size(hdim)},
            SizeArray{dtmp.size(hdim), dtmp.size(wdim)}
            );
    }
    else{
        HMP_REQUIRE(stmp.size(wdim) == dtmp.size(wdim) && stmp.size(hdim) == dtmp.size(hdim), 
            "img_rotate: image size is invalid, expec {}, got {}", 
            SizeArray{stmp.size(wdim), stmp.size(hdim)},
            SizeArray{dtmp.size(wdim), dtmp.size(hdim)}
            );
    }

    img_rotate_stub(dtmp.device_type(), dtmp, stmp, mode, cformat);

    return dst;
}


Tensor &img_mirror(Tensor &dst, const Tensor &src, ImageAxis axis, ChannelFormat cformat)
{
    auto stmp = img::image_format(src, cformat);
    auto dtmp = img::image_format(dst, cformat);

    img_common_check(dtmp, stmp, cformat, "img_mirror");
    HMP_REQUIRE(stmp.shape() == dtmp.shape(), 
        "img_mirror: expect src and dst have same shape, got src={}, dst={}", stmp.shape(), dtmp.shape());

    img_mirror_stub(stmp.device_type(), dtmp, stmp, axis, cformat);

    return dst;
}


Tensor& img_normalize(Tensor &dst, const Tensor &src, const Tensor &mean, const Tensor &std, ChannelFormat cformat)
{
    auto stmp = img::image_format(src, cformat);
    auto dtmp = img::image_format(dst, cformat);

    checkDevice({stmp, mean, std}, stmp.device(), "img_normalize");
    img_common_check(dtmp, stmp, cformat, "img_normalize");
    HMP_REQUIRE(stmp.shape() == dtmp.shape(), 
        "img_normalize: expect src and dst have same shape, got src={}, dst={}", stmp.shape(), dtmp.shape());
    auto cdim = cformat == kNCHW ? 1 : -1;
    HMP_REQUIRE(mean.dim() == 1 && std.dim() == 1 && 
            mean.size(0) == std.size(0) && mean.size(0) == stmp.size(cdim),
        "img_normalize: invalid mean or std shape, expect ({},)", stmp.size(cdim));

    img_normalize_stub(stmp.device_type(), dtmp, stmp, mean, std, cformat);

    return dst;

}


Tensor &img_erode(Tensor &dst, const Tensor &src, const Tensor &kernel, ChannelFormat cformat)
{
    auto stmp = img::image_format(src, cformat);
    auto dtmp = img::image_format(dst, cformat);

    img_common_check(dtmp, stmp, cformat, "img_erode");
    HMP_REQUIRE(stmp.shape() == dtmp.shape(), 
        "img_erode: expect src and dst have same shape, got src={}, dst={}", stmp.shape(), dtmp.shape());
    img_erode_stub(stmp.device_type(), dtmp, stmp, kernel, cformat);

    return dst;
}

Tensor &img_dilate(Tensor &dst, const Tensor &src, const Tensor &kernel, ChannelFormat cformat)
{
    auto stmp = img::image_format(src, cformat);
    auto dtmp = img::image_format(dst, cformat);

    img_common_check(dtmp, stmp, cformat, "img_dilate");
    HMP_REQUIRE(stmp.shape() == dtmp.shape(), 
        "img_dilate: expect src and dst have same shape, got src={}, dst={}", stmp.shape(), dtmp.shape());
    img_dilate_stub(stmp.device_type(), dtmp, stmp, kernel, cformat);

    return dst;
}


Tensor &img_sobel(Tensor &dst, const Tensor &src, int64_t dx, int64_t dy,
    int64_t ksize, const Scalar& scale, const Scalar& delta, ChannelFormat cformat)
{
    auto stmp = img::image_format(src, cformat);
    auto dtmp = img::image_format(dst, cformat);

    img_common_check(dtmp, stmp, cformat, "img_sobel");
    HMP_REQUIRE(stmp.shape() == dtmp.shape(), 
        "img_sobel: expect src and dst have same shape, got src={}, dst={}", stmp.shape(), dtmp.shape());

    img_sobel_stub(stmp.device_type(), dtmp, stmp, dx, dy, ksize, scale, delta, cformat);

    return dst;
}


Tensor &img_canny(Tensor &dst, const Tensor &src, const Scalar &low_thresh, const Scalar &high_thresh,
    int64_t aperture_size, bool l2_gradient, ChannelFormat cformat)
{
    auto stmp = img::image_format(src, cformat);
    auto dtmp = img::image_format(dst, cformat);

    img_common_check(dtmp, stmp, cformat, "img_canny");
    auto cdim = cformat == kNCHW ? 1 : -1;
    HMP_REQUIRE(dtmp.size(cdim) == 1,
                "img_canny: invalid dst shape, expect 1 channel, got {}", stmp.size(cdim));

    img_canny_stub(stmp.device_type(), dtmp, stmp, low_thresh, high_thresh,
         aperture_size, l2_gradient, cformat);

    return dst;
}


Tensor &img_warp_perspective(Tensor &dst, const Tensor &src,
    const Tensor &M, ImageFilterMode mode, ChannelFormat cformat)
{
    auto stmp = img::image_format(src, cformat);
    auto dtmp = img::image_format(dst, cformat);

    img_common_check(dtmp, stmp, cformat, "img_warp_perspective");

    img_warp_perspective_stub(stmp.device_type(), dtmp, stmp, M,
        mode, cformat);

    return dst;
}


Tensor &img_filter2d(Tensor &dst, const Tensor &src,
    const Tensor &kernel, const Scalar& delta, ChannelFormat cformat)
{
    auto stmp = img::image_format(src, cformat);
    auto dtmp = img::image_format(dst, cformat);

    img_common_check(dtmp, stmp, cformat, "img_filter2d");
    HMP_REQUIRE(stmp.shape() == dtmp.shape(), 
        "img_filter2d: expect src and dst have same shape, got src={}, dst={}", stmp.shape(), dtmp.shape());
    img_filter2d_stub(stmp.device_type(), dtmp, stmp, kernel, delta, cformat);

    return dst;
}


Tensor& img_bilateral_filter(Tensor &dst, const Tensor &src, int d, 
    const Scalar& sigma_color, const Scalar& sigma_space, ChannelFormat cformat)
{
    auto stmp = img::image_format(src, cformat);
    auto dtmp = img::image_format(dst, cformat);

    img_common_check(dtmp, stmp, cformat, "img_bilateral_filter");
    HMP_REQUIRE(stmp.shape() == dtmp.shape(), 
        "img_bilateral_filter: expect src and dst have same shape, got src={}, dst={}", stmp.shape(), dtmp.shape());
    img_bilateral_filter_stub(stmp.device_type(), dtmp, stmp, d, sigma_color, sigma_space, cformat);

    return dst;
}

Tensor& img_gaussian_blur(Tensor &dst, const Tensor &src, int kx, int ky,
     const Scalar& sigma_x, const Scalar& sigma_y, ChannelFormat cformat)
{
    auto stmp = img::image_format(src, cformat);
    auto dtmp = img::image_format(dst, cformat);

    img_common_check(dtmp, stmp, cformat, "img_gaussian_blur");
    HMP_REQUIRE(stmp.shape() == dtmp.shape(), 
        "img_gaussian_blur: expect src and dst have same shape, got src={}, dst={}", stmp.shape(), dtmp.shape());
    img_gaussian_blur_stub(stmp.device_type(), dtmp, stmp, kx, ky, sigma_x, sigma_y, cformat);

    return dst;
}


Tensor &img_overlay(Tensor &dst, const Tensor &src0, const Tensor &src1,
    const Tensor &alpha, ChannelFormat cformat)
{
    HMP_REQUIRE(dst.dim() == 4,
            "img_overlay: expect dst tensor have 4 dims, got {}", dst.dim());
    checkShape({src0, src1}, dst.shape(), "img_overlay");
    HMP_REQUIRE(alpha.dim() >= 2,
            "img_overlay: expect alpha tensor have at least 2 dims, got {}", alpha.dim());

    //
    auto tmp = alpha;
    auto new_shape = tmp.shape();
    if(new_shape.size() == 2){
        new_shape.insert(new_shape.begin(), 1); //unsqueeze batch dim
    }
    if(new_shape.size() == 3){ //unsqueeze channel dim
        if(cformat == kNCHW){
            new_shape.insert(new_shape.begin()+1, 1);
        }
        else{
            new_shape.insert(new_shape.end(), 1);
        }
    }
    tmp = tmp.view(new_shape).expand_as(dst);

    //
    if(!dst.is_same(src0)){
        copy(dst, src0);
    }

    dst.mul_(1-tmp).add_(tmp * src1);

    return dst;
}



}} //namespace