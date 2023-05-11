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
#include <kernel/imgproc.h>
#include <hmp/format.h>

namespace hmp{
namespace img{


TensorList frame_format(const TensorList &data, const PixelFormatDesc &pix_desc, int width, int height, bool has_batch)
{
    HMP_REQUIRE(data.size() == pix_desc.nplanes(), 
        "Expect {} planes for pixel format {}, got {}",
         pix_desc.nplanes(), pix_desc.format(), data.size());

    TensorList out;
    for(int i = 0; i < data.size(); ++i){
        SizeArray shape{pix_desc.infer_height(height, i),
                        pix_desc.infer_width(width, i),
                        pix_desc.channels(i)};
        if(has_batch){
            shape.insert(shape.begin(), data[i].size(0));
        }
        HMP_REQUIRE(data[i].dtype() == pix_desc.dtype(),
            "Expect {} for pixel format {}, got {}", 
            pix_desc.dtype(), pix_desc.format(), data[i].dtype());

        out.push_back(data[i].view(shape));
    }
    return out;
}

TensorList frame_format(const TensorList &data, PixelFormat format, int width, int height, bool has_batch)
{
    return frame_format(data, PixelFormatDesc(format), width, height, has_batch);
}


TensorList frame_format(const TensorList &data, PixelFormat format, bool has_batch)
{
    HMP_REQUIRE(data[0].dim() >= 2 + has_batch, 
        "Infer frame size failed, expect ndim >= {}, got {}",
         2 + has_batch, data[0].dim());

    int hdim = has_batch ? 1 : 0;
    int width = data[0].size(hdim+1);
    int height = data[0].size(hdim);
    return frame_format(data, format, width, height, has_batch);
}


Tensor image_format(const Tensor &image, ChannelFormat cformat, bool batch_first)
{
    if(image.dim() == 4){
        return image;
    }
    else if(image.dim() == 3){ //add batch dim
        if(batch_first){
            return image.unsqueeze(0);
        }
        else{
            if(cformat == kNCHW){
                return image.unsqueeze(1);
            }
            else{
                return image.unsqueeze(3);
            }
        }
    }
    else if(image.dim() == 2){
        if(cformat == kNCHW){
            return image.unsqueeze(0).unsqueeze(1);
        }
        else{
            return image.unsqueeze(0).unsqueeze(3);
        }
    }
    else{
        HMP_REQUIRE(false, "Image data need at least 2 dims and less than or equal to 4 dims, got {}",
            image.dim());
    }
}


TensorList image_format(const TensorList &images, ChannelFormat cformat, bool batch_first)
{
    TensorList vimages;
    for(auto &img : images){
        vimages.push_back(image_format(img, cformat, batch_first));
    }
    return vimages;
}

#ifndef HMP_ENABLE_MOBILE


static SizeArray remove_cdim(const Tensor &src, ChannelFormat cformat)
{
    SizeArray shape;
    if(src.dim() >= 3){
        if(cformat == kNCHW){
            shape = {1, src.size(-2), src.size(-1)};
        }
        else{
            shape = {src.size(-3), src.size(-2), 1};
        }
    }
    if(src.dim() == 4){
        shape.insert(shape.begin(), src.size(0));
    }

    return shape;
} 


static int infer_wdim(const Tensor &im, ChannelFormat cformat)
{
    HMP_REQUIRE(im.dim() >= 2, "Image need at least 2 dims, got {}", im.dim());
    return (cformat == kNCHW || im.dim() == 2) ? im.dim() - 1 : im.dim() - 2;
}


static PPixelFormat infer_ppixel_format(const PixelInfo &info)
{
    auto space = info.infer_space();
    if(space == CS_BT709){
        switch(info.format()){
            case PF_YUV420P:
                return PPixelFormat::H420;
            case PF_YUV422P:
                return PPixelFormat::H422;
            case PF_YUV444P:
                return PPixelFormat::H444;
            case PF_NV12:
                return PPixelFormat::NV12_BT709;
            case PF_NV21:
                return PPixelFormat::NV21_BT709;
            default:
                HMP_REQUIRE(false, "Unsupport PixelInfo");
        }
    }
    else if(space == CS_BT470BG){
        switch(info.format()){
            case PF_YUV420P:
                return PPixelFormat::I420;
            case PF_YUV422P:
                return PPixelFormat::I422;
            case PF_YUV444P:
                return PPixelFormat::I444;
            case PF_NV12:
                return PPixelFormat::NV12;
            case PF_NV21:
                return PPixelFormat::NV21;
            default:
                HMP_REQUIRE(false, "Unsupport PixelInfo");
        }
    }
    HMP_REQUIRE(false, "Unsupport PixelInfo");
}


Tensor &yuv_to_rgb(Tensor &dst, const TensorList &src, const PixelInfo &pix_info, ChannelFormat cformat)
{
    auto pformat = infer_ppixel_format(pix_info);
    return kernel::yuv_to_rgb(dst, src, pformat, cformat);
}

Tensor yuv_to_rgb(const TensorList &src, const PixelInfo &pix_info, ChannelFormat cformat)
{
    auto has_batch_dim = src[0].dim() == 4;
    auto stmp = frame_format(src, pix_info.format(), has_batch_dim);
    
    auto dshape = SizeArray(4, 0);
    auto batch = has_batch_dim ? stmp[0].size(0) : int64_t(1);
    auto width = has_batch_dim ? stmp[0].size(2) : stmp[0].size(1);
    auto height = has_batch_dim ? stmp[0].size(1) : stmp[0].size(0);
    if(cformat == ChannelFormat::NCHW){
        dshape = {batch, int64_t(3), height, width};
    }
    else{
        dshape = {batch, height, width, int64_t(3)};
    }
    auto dst = empty(dshape, stmp[0].options());
    dst = yuv_to_rgb(dst, src, pix_info, cformat);

    if(!has_batch_dim){
        dst.squeeze_(0);
    }
    return dst;
}

TensorList &rgb_to_yuv(TensorList &dst, const Tensor &src, const PixelInfo &pix_info, ChannelFormat cformat)
{
    auto pformat = infer_ppixel_format(pix_info);
    return kernel::rgb_to_yuv(dst, src, pformat, cformat);
}


TensorList rgb_to_yuv(const Tensor &src, const PixelInfo &pix_info, ChannelFormat cformat)
{
    auto wdim = infer_wdim(src, cformat);
    auto hdim = wdim - 1;
    auto pix_desc = PixelFormatDesc(pix_info.format());

    TensorList dst;
    auto has_batch_dim = src.dim() == 4;
    for(int i = 0; i < pix_desc.nplanes(); ++i){
        auto width = pix_desc.infer_width(src.size(wdim), i);
        auto height = pix_desc.infer_height(src.size(hdim), i);
        auto channels = pix_desc.channels(i);
        if(has_batch_dim){
            dst.push_back(empty({src.size(0), height, width, channels}, src.options()));
        }
        else{
            dst.push_back(empty({height, width, channels}, src.options()));
        }
    }

    return rgb_to_yuv(dst, src, pix_info, cformat);
}


TensorList &yuv_resize(TensorList &dst, const TensorList &src,
                          const PixelInfo &pix_info, ImageFilterMode mode)
{
    auto format = infer_ppixel_format(pix_info);
    return kernel::yuv_resize(dst, src, format, mode);
}

TensorList &yuv_rotate(TensorList &dst, const TensorList &src,
                           const PixelInfo &pix_info, ImageRotationMode rotate)
{
    auto format = infer_ppixel_format(pix_info);
    return kernel::yuv_rotate(dst, src, format, rotate);
}

TensorList &yuv_mirror(TensorList &dst, const TensorList &src, const PixelInfo &pix_info, ImageAxis axis)
{
    auto format = infer_ppixel_format(pix_info);
    return kernel::yuv_mirror(dst, src, format, axis);
}

Tensor &resize(Tensor &dst, const Tensor &src, ImageFilterMode mode, ChannelFormat cformat)
{
    return kernel::img_resize(dst, src, mode, cformat);
}

Tensor resize(const Tensor &src, int width, int height, ImageFilterMode mode, ChannelFormat cformat)
{
    auto dshape = src.shape();
    auto wdim = (cformat == kNCHW || src.dim() == 2) ? src.dim() - 1 : src.dim() - 2;
    auto hdim = wdim - 1;
    dshape[wdim] = width;
    dshape[hdim] = height;
    auto dst = empty(dshape, src.options());

    return resize(dst, src, mode, cformat);
}

Tensor &rotate(Tensor &dst, const Tensor &src, ImageRotationMode mode, ChannelFormat cformat)
{
    return kernel::img_rotate(dst, src, mode, cformat);
}

Tensor rotate(const Tensor &src, ImageRotationMode mode, ChannelFormat cformat)
{
    auto dshape = src.shape();
    auto wdim = (cformat == kNCHW || src.dim() == 2) ? src.dim() - 1 : src.dim() - 2;
    auto hdim = wdim - 1;
    if(mode == ImageRotationMode::Rotate90 || mode == ImageRotationMode::Rotate270){
        std::swap(dshape[wdim], dshape[hdim]);
    }
    auto dst = empty(dshape, src.options());

    return rotate(dst, src, mode, cformat);
}

Tensor &mirror(Tensor &dst, const Tensor &src, ImageAxis axis, ChannelFormat cformat)
{
    return kernel::img_mirror(dst, src, axis, cformat);
}


Tensor mirror(const Tensor &src, ImageAxis axis, ChannelFormat cformat)
{
    auto dst = empty_like(src);
    return mirror(dst, src, axis, cformat);
}

Tensor normalize(const Tensor &src, const Tensor &mean, const Tensor& std, ChannelFormat cformat)
{
    auto dst = empty_like(src, src.options().dtype(kFloat32));
    return normalize(dst, src, mean, std, cformat);
}

Tensor& normalize(Tensor &dst, const Tensor &src, const Tensor &mean, const Tensor &std, ChannelFormat cformat)
{
    return kernel::img_normalize(dst, src, mean, std, cformat);
}

Tensor &erode(Tensor &dst, const Tensor &src, const optional<Tensor> &kernel_, ChannelFormat cformat)
{
    Tensor kernel;
    if(kernel_){
        kernel = kernel_.value();
    }
    else{
        kernel = ones({3, 3}, src.options().dtype(kFloat32));
    }

    return kernel::img_erode(dst, src, kernel, cformat);
}


Tensor erode(const Tensor &src, const optional<Tensor> &kernel_, ChannelFormat cformat)
{
    auto dst = empty_like(src);
    return erode(dst, src, kernel_, cformat);
}


Tensor &dilate(Tensor &dst, const Tensor &src, const optional<Tensor> &kernel_, ChannelFormat cformat)
{
    Tensor kernel;
    if(kernel_){
        kernel = kernel_.value();
    }
    else{
        kernel = ones({3, 3}, src.options().dtype(kFloat32));
    }
    return kernel::img_dilate(dst, src, kernel, cformat);
}

Tensor dilate(const Tensor &src, const optional<Tensor> &kernel_, ChannelFormat cformat)
{
    auto dst = empty_like(src);
    return dilate(dst, src, kernel_, cformat);

}

Tensor &sobel(Tensor &dst, const Tensor &src, int64_t dx, int64_t dy,
    int64_t ksize, const Scalar& scale, const Scalar& delta,
    ChannelFormat cformat)
{
    return kernel::img_sobel(dst, src, dx, dy, ksize, scale, delta, cformat);
}


Tensor sobel(const Tensor &src, int64_t dx, int64_t dy,
    int64_t ksize, const Scalar& scale, const Scalar& delta,
    ChannelFormat cformat)
{
    auto dst = empty_like(src);
    return sobel(dst, src, dx, dy, ksize, scale, delta, cformat);
}

Tensor &canny(Tensor &dst, const Tensor &src, const Scalar &low_thresh, const Scalar &high_thresh,
    int64_t aperture_size, bool l2_gradient, ChannelFormat cformat)
{
    return kernel::img_canny(
        dst, src, low_thresh, high_thresh, aperture_size, l2_gradient, cformat);
}

Tensor canny(const Tensor &src, const Scalar &low_thresh, const Scalar &high_thresh,
    int64_t aperture_size, bool l2_gradient, ChannelFormat cformat)
{
    auto shape = remove_cdim(src, cformat);
    auto dst = empty(shape, src.options());
    canny(dst, src, low_thresh, high_thresh, aperture_size, l2_gradient, cformat);
    dst.squeeze_();
    return dst;
}


Tensor &filter2d(Tensor &dst, const Tensor &src,
    const Tensor &kernel, const Scalar& delta, ChannelFormat cformat)
{
    return kernel::img_filter2d(dst, src, kernel, delta, cformat);
}


Tensor filter2d(const Tensor &src,
    const Tensor &kernel, const Scalar& delta, ChannelFormat cformat)
{
    auto dst = empty_like(src);
    return filter2d(dst, src, kernel, delta, cformat);
}


Tensor bilateral_filter(const Tensor &src, int d, 
    const Scalar& sigma_color, const Scalar& sigma_space, ChannelFormat cformat)
{
    auto dst = empty_like(src);
    return bilateral_filter(dst, src, d, sigma_color, sigma_space, cformat);
}

Tensor& bilateral_filter(Tensor &dst, const Tensor &src, int d, 
    const Scalar& sigma_color, const Scalar& sigma_space, ChannelFormat cformat)
{
    return kernel::img_bilateral_filter(dst, src, d, sigma_color, sigma_space, cformat);
}

Tensor gaussian_blur(const Tensor &src, int kx, int ky,
     const Scalar& sigma_x, const Scalar& sigma_y, ChannelFormat cformat)
{
    auto dst = empty_like(src);
    return gaussian_blur(dst, src, kx, ky, sigma_x, sigma_y, cformat);
}

Tensor& gaussian_blur(Tensor &dst, const Tensor &src, int kx, int ky,
     const Scalar& sigma_x, const Scalar& sigma_y, ChannelFormat cformat)
{
    return kernel::img_gaussian_blur(dst, src, kx, ky, sigma_x, sigma_y, cformat);
}


Tensor warp_perspective(const Tensor &src, int64_t width, int64_t height,
    const Tensor &M, ImageFilterMode mode, ChannelFormat cformat)
{
    auto wdim = infer_wdim(src, cformat);
    auto dshape = src.shape();
    dshape[wdim] = width;
    dshape[wdim-1] = height;
    auto dst = empty(dshape, src.options());

    warp_perspective(dst, src, M, mode, cformat);

    return dst;
}

Tensor &warp_perspective(Tensor &dst, const Tensor &src,
    const Tensor &M, ImageFilterMode mode, ChannelFormat cformat)
{
    return kernel::img_warp_perspective(dst, src, M, mode, cformat);
}


//dst = src0 * (1 - alpha) + src1 * (alpha);
Tensor &overlay(Tensor &dst, const Tensor &src0, const Tensor &src1, const Tensor &alpha)
{
    return kernel::img_overlay(dst, src0, src1, alpha);
}


Tensor overlay( const Tensor &src0, const Tensor &src1, const Tensor &alpha)
{
    auto dst = empty_like(src0);
    return overlay(dst, src0, src1, alpha);
}

Tensor transfer(const Tensor &src, const ChannelFormat &src_format, const ChannelFormat &dst_format)
{
    HMP_REQUIRE(src.dim() == 3 || src.dim() == 4, "dims must be 3 or 4");

    Tensor dst;
    if (src.dim() == 3) {
        if(dst_format == ChannelFormat::NCHW && src_format == ChannelFormat::NHWC){
            dst = src.permute({2, 0, 1});
        } else if(dst_format == ChannelFormat::NHWC && src_format == ChannelFormat::NCHW){
            dst = src.permute({1, 2, 0});
        } else {
            dst = src;
        }
        dst = dst.contiguous();

    } else if (src.dim() == 4) {
        if(dst_format == ChannelFormat::NCHW && src_format == ChannelFormat::NHWC){
            dst = src.permute({0, 3, 1, 2});
        } else if(dst_format == ChannelFormat::NHWC && src_format == ChannelFormat::NCHW){
            dst = src.permute({0, 2, 3, 1});
        } else {
            dst = src;
        }
        dst = dst.contiguous();
    }
    return dst;
}

#endif //HMP_ENABLE_MOBILE


}} //namespace hmp::img
