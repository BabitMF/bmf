

#include <type_traits>
#include <kernel/imgproc.h>
#include <kernel/kernel_utils.h>
#include <kernel/image_iter.h>
#include <kernel/image_filter.h>
#include <kernel/image_color_cvt.h>
#include <kernel/cuda/kernel_utils.h>

namespace hmp{
namespace kernel{
namespace{



// scalar_t, dst, src, batch, width, height need pre-defined
#define PIXEL_FORMAT_CASE(Op, Format, Cformat)                                                  \
    case(PPixelFormat::Format):                                                                  \
        do{                                                                                     \
            Op<scalar_t, PPixelFormat::Format, Cformat> op(dst, src);                            \
            cuda::invoke_img_elementwise_kernel([=]HMP_HOST_DEVICE(int batch, int w, int h) mutable{\
                op(batch, w, h);                                                                \
            }, batch, width, height);                                                           \
        }while(0);                                                                              \
        break;


#define PIXEL_FORMAT_DISPATCH(Op, format, Cformat, name)                                        \
    switch(format){                                                                             \
        PIXEL_FORMAT_CASE(Op, H420, Cformat)                                                   \
        PIXEL_FORMAT_CASE(Op, H422, Cformat)                                                   \
        PIXEL_FORMAT_CASE(Op, H444, Cformat)                                                   \
        PIXEL_FORMAT_CASE(Op, I420, Cformat)                                                   \
        PIXEL_FORMAT_CASE(Op, I422, Cformat)                                                   \
        PIXEL_FORMAT_CASE(Op, I444, Cformat)                                                   \
        PIXEL_FORMAT_CASE(Op, NV21, Cformat)                                                   \
        PIXEL_FORMAT_CASE(Op, NV12, Cformat)                                                   \
        default:                                                                                \
            HMP_REQUIRE(false, "{} : unsupported PPixelFormat {}", name, format);                \
     }


Tensor &yuv_to_rgb_cuda(Tensor &dst, const TensorList &src, PPixelFormat format, ChannelFormat cformat)
{
    auto batch = src[0].size(0);
    auto height = src[0].size(1);
    auto width = src[0].size(2);

    HMP_DISPATCH_IMAGE_TYPES_AND_HALF(src[0].scalar_type(), "yuv_to_rgb_cuda", [&](){
        if(cformat == kNCHW){
            PIXEL_FORMAT_DISPATCH(YUV2RGB, format, kNCHW, "yuv_to_rgb_cuda");
        }
        else{
            PIXEL_FORMAT_DISPATCH(YUV2RGB, format, ChannelFormat::NHWC, "yuv_to_rgb_cuda");
        }
    });
 
    return dst;
}



TensorList &rgb_to_yuv_cuda(TensorList &dst, const Tensor &src, PPixelFormat format, ChannelFormat cformat)
{
    auto batch = dst[0].size(0);
    auto height = dst[0].size(1);
    auto width = dst[0].size(2);

    HMP_DISPATCH_IMAGE_TYPES_AND_HALF(dst[0].scalar_type(), "rgb_to_yuv_cuda", [&](){
        if(cformat == kNCHW){
            PIXEL_FORMAT_DISPATCH(RGB2YUV, format, kNCHW, "rgb_to_yuv_cuda");
        }
        else{
            PIXEL_FORMAT_DISPATCH(RGB2YUV, format, ChannelFormat::NHWC, "rgb_to_yuv_cuda");
        }
    });
 
    return dst;
}


Tensor &img_resize_cuda(Tensor &dst, const Tensor &src, ImageFilterMode mode, ChannelFormat cformat)
{
    HMP_DISPATCH_IMAGE_TYPES_AND_HALF(src.scalar_type(), "img_resize_cuda", [&](){
        auto channel = cformat == kNCHW ? 1 : src.size(-1);
        HMP_DISPATCH_IMAGE_CHANNEL(channel, "img_resize_cuda", [&](){
            using vtype = Vector<scalar_t, C::size()>;
            using wtype = Vector<float, C::size()>;

            using Iter = ImageSeqIter<vtype>;
            auto src_iter = Iter::from_tensor(src, cformat);
            auto dst_iter = Iter::from_tensor(dst, cformat);

            auto wscale = float(src_iter.width_) / dst_iter.width_;
            auto hscale = float(src_iter.height_) / dst_iter.height_;
            auto wscale_offset = 0.f;
            auto hscale_offset = 0.f;
            if(mode == ImageFilterMode::Bilinear || mode == ImageFilterMode::Bicubic){
                wscale_offset = 0.5f * wscale - 0.5f;
                hscale_offset = 0.5f * hscale - 0.5f;
            }

            HMP_DISPATCH_IMAGE_FILTER(mode, Iter, wtype, vtype, "img_resize_cuda", [&](){
                filter_t filter(src_iter); 
                cuda::invoke_img_elementwise_kernel([=] HMP_HOST_DEVICE(int batch, int w, int h) mutable {
                    auto x = w * wscale + wscale_offset;
                    auto y = h * hscale + hscale_offset; 
                    dst_iter.set(batch, w, h, filter(batch, x, y));
                }, dst_iter.batch_, dst_iter.width_, dst_iter.height_, 64, 4);
            });
        });
    });

    return dst;
}


TensorList &yuv_resize_cuda(TensorList &dst, const TensorList &src,
                          PPixelFormat format, ImageFilterMode mode)
{
    for(size_t i = 0; i < src.size(); ++i){
        img_resize_cuda(dst[i], src[i], mode, ChannelFormat::NHWC);
    }

    return dst;
}



Tensor &img_mirror_cuda(Tensor &dst, const Tensor &src, ImageAxis axis, ChannelFormat cformat)
{
    HMP_DISPATCH_IMAGE_TYPES_AND_HALF(src.scalar_type(), "img_mirror_cuda", [&](){
        auto channel = cformat == kNCHW ? 1 : src.size(-1);
        HMP_DISPATCH_IMAGE_CHANNEL(channel, "img_mirror_cuda", [&](){
            using vtype = Vector<scalar_t, C::size()>;

            using Iter = ImageSeqIter<vtype>;
            auto src_iter = Iter::from_tensor(src, cformat);
            auto dst_iter = Iter::from_tensor(dst, cformat);

            cuda::invoke_img_elementwise_kernel([=] HMP_HOST_DEVICE(int batch, int w, int h) mutable {
                auto hmirror = static_cast<uint8_t>(axis) & static_cast<uint8_t>(ImageAxis::Horizontal);
                auto vmirror = static_cast<uint8_t>(axis) & static_cast<uint8_t>(ImageAxis::Vertical);
                auto x = hmirror ? dst_iter.width_- 1 - w : w; 
                auto y = vmirror ? dst_iter.height_- 1 - h : h; 
                dst_iter.set(batch, w, h, src_iter.get(batch, x, y));
            }, dst_iter.batch_, dst_iter.width_, dst_iter.height_);
        });

    });

    return dst;
}


TensorList &yuv_mirror_cuda(TensorList &dst, const TensorList &src,
                           PPixelFormat format, ImageAxis axis)
{
    for(size_t i = 0; i < src.size(); ++i){
        img_mirror_cuda(dst[i], src[i], axis, ChannelFormat::NHWC);
    }

    return dst;
}


Tensor &img_rotate_cuda(Tensor &dst, const Tensor &src, ImageRotationMode mode, ChannelFormat cformat)
{
    HMP_DISPATCH_IMAGE_TYPES_AND_HALF(src.scalar_type(), "img_rotate_cuda", [&](){
        auto channel = cformat == kNCHW ? 1 : src.size(-1);
        HMP_DISPATCH_IMAGE_CHANNEL(channel, "img_rotate_cuda", [&](){
            using vtype = Vector<scalar_t, C::size()>;

            using Iter = ImageSeqIter<vtype>;
            auto src_iter = Iter::from_tensor(src, cformat);
            auto dst_iter = Iter::from_tensor(dst, cformat);

            cuda::invoke_img_elementwise_kernel([=] HMP_HOST_DEVICE(int batch, int w, int h) mutable {
                int x, y;
                switch(mode){
                    case ImageRotationMode::Rotate90:
                        x = h;
                        y = dst_iter.width_ - 1 - w;
                        break;
                    case ImageRotationMode::Rotate180:
                        x = dst_iter.width_ - 1 - w;
                        y = dst_iter.height_ - 1 - h;
                        break;
                    case ImageRotationMode::Rotate270:
                        x = dst_iter.height_ - 1 - h;
                        y = w;
                        break;
                    default:
                        x = w;
                        y = h;
                }
                dst_iter.set(batch, w, h, src_iter.get(batch, x, y));

            }, dst_iter.batch_, dst_iter.width_, dst_iter.height_);
        });

    });

    return dst;
}


TensorList &yuv_rotate_cuda(TensorList &dst, const TensorList &src,
                           PPixelFormat format, ImageRotationMode mode)
{
    for(size_t i = 0; i < src.size(); ++i){
        img_rotate_cuda(dst[i], src[i], mode, ChannelFormat::NHWC);
    }

    return dst;
}


Tensor& img_normalize_cuda(Tensor &dst, const Tensor &src, const Tensor &mean, const Tensor &std, ChannelFormat cformat)
{
    auto fmean = mean.to(kFloat32);
    auto fstd = std.to(kFloat32);

    HMP_DISPATCH_IMAGE_TYPES_AND_HALF(src.scalar_type(), "img_normalize_cuda", [&](){
        using iscalar_t = scalar_t;
        HMP_DISPATCH_FLOAT32_AND_HALF(dst.scalar_type(), "img_normalize_cuda", [&](){
            using oscalar_t = scalar_t;
            HMP_DISPATCH_CHANNEL_FORMAT(cformat, "img_normalize_cuda", [&](){
                int channel = cformat == kNCHW ? src.size(1) : src.size(-1);
                HMP_DISPATCH_IMAGE_CHANNEL(channel, "img_normalize_cuda", [&](){
                    using itype = Vector<iscalar_t, C::size()>;
                    using otype = Vector<oscalar_t, C::size()>;
                    auto src_iter = ImageSeqIter<itype, FMT>::from_tensor(src, cformat);
                    auto dst_iter = ImageSeqIter<otype, FMT>::from_tensor(dst, cformat);
                    auto fmean_ptr = fmean.data<float>();
                    auto fmean_stride = fmean.stride(0);
                    auto fstd_ptr = fstd.data<float>();
                    auto fstd_stride = fstd.stride(0);

                    cuda::invoke_img_elementwise_kernel([=]HMP_HOST_DEVICE(int batch, int w, int h) mutable {
                        auto in = src_iter.get(batch, w, h);
                        otype out;
                        for(int i = 0; i < C::size(); ++i){
                            out[i] = (in[i] - fmean_ptr[i*fmean_stride]) / fstd_ptr[i*fstd_stride];
                        }
                        dst_iter.set(batch, w, h, out);
                    }, dst_iter.batch_, dst_iter.width_, dst_iter.height_);
                });
            });
        });
    });


    return dst;
}


HMP_DEVICE_DISPATCH(kCUDA, yuv_to_rgb_stub, &yuv_to_rgb_cuda)
HMP_DEVICE_DISPATCH(kCUDA, rgb_to_yuv_stub, &rgb_to_yuv_cuda)
HMP_DEVICE_DISPATCH(kCUDA, yuv_resize_stub, &yuv_resize_cuda)
HMP_DEVICE_DISPATCH(kCUDA, yuv_mirror_stub, &yuv_mirror_cuda)
HMP_DEVICE_DISPATCH(kCUDA, yuv_rotate_stub, &yuv_rotate_cuda)
HMP_DEVICE_DISPATCH(kCUDA, img_resize_stub, &img_resize_cuda)
HMP_DEVICE_DISPATCH(kCUDA, img_mirror_stub, &img_mirror_cuda)
HMP_DEVICE_DISPATCH(kCUDA, img_rotate_stub, &img_rotate_cuda)
HMP_DEVICE_DISPATCH(kCUDA, img_normalize_stub, &img_normalize_cuda)

}}} //namespace hmp::kernel