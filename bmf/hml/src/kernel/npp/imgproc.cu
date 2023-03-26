#include <kernel/imgproc.h>
#include <kernel/kernel_utils.h>
#include <kernel/cuda/kernel_utils.h>
#include <kernel/npp/npp_wrapper.h>

namespace hmp{
namespace kernel{
namespace{


Tensor &yuv_to_rgb_cuda(Tensor &dst, const TensorList &src, PPixelFormat pformat, ChannelFormat cformat)
{
    auto w = dst.scalar_type() == kUInt8 ? dst : empty_like(dst, dst.options().dtype(kUInt8));
    int width = w.size(2);
    int height = w.size(1);

    if(pformat == PPixelFormat::H420 || pformat == PPixelFormat::I420){
        HMP_REQUIRE(src.size() == 3, "yuv_to_rgbx_cuda: expect 3 planes, got {}", src.size());
        auto streamCtx = makeNppStreamContext();
        auto batch = dst.size(0);
        for(int64_t i = 0; i < batch; ++i){
            auto y = src[0].select(0, i);
            auto u = src[1].select(0, i);
            auto v = src[2].select(0, i);
            auto rgb = dst.select(0, i);

            const Npp8u *pSrc[3]{y.data<uint8_t>(), u.data<uint8_t>(), v.data<uint8_t>()};
            int srcStep[3]{(int)y.stride(0), (int)u.stride(0), (int)v.stride(0)};
            Npp8u *pDst = rgb.data<uint8_t>();
            int dstStep = rgb.stride(0);
            
            //[NOTE] batch call need copy NppiImageDescriptor to device
            auto func = nppiYCbCr420ToRGB_8u_P3C3R_Ctx;
            if(pformat == PixelFormat::I420){
                func = nppiYUV420ToRGB_8u_P3C3R_Ctx;
            }

            auto status = func(pSrc, srcStep, pDst, dstStep, NppiSize{width, height}, streamCtx);
            HMP_REQUIRE(status == NPP_SUCCESS, 
                "nppiXXX420ToRGB_8u_P3C3R_Ctx failed with status={}", status);
        }
    }
    else{
        HMP_REQUIRE(false, "unsupport pixel format", pformat);
    }

    if(dst.scalar_type() != kUInt8){
        copy(dst, w);
    }

    return dst;
}


TensorList &rgb_to_yuv_cuda(TensorList &dst, const Tensor &src, PixelFormat pformat, ChannelFormat cformat)
{
    int width = src.size(2);
    int height = src.size(1);

    if(pformat == PixelFormat::H420 || pformat == PixelFormat::I420){
        HMP_REQUIRE(dst.size() == 3, "rgbx_to_yuv_cuda: expect 3 planes, got {}", dst.size());
        auto streamCtx = makeNppStreamContext();
        auto batch = src.size(0);
        for(int64_t i = 0; i < batch; ++i){
            auto y = dst[0].select(0, i);
            auto u = dst[1].select(0, i);
            auto v = dst[2].select(0, i);
            auto rgb = src.select(0, i);

            Npp8u *pDst[3]{y.data<uint8_t>(), u.data<uint8_t>(), v.data<uint8_t>()};
            int dstStep[3]{(int)y.stride(0), (int)u.stride(0), (int)v.stride(0)};
            const Npp8u *pSrc = rgb.data<uint8_t>();
            int srcStep = rgb.stride(0);
            
            //[NOTE] batch call need copy NppiImageDescriptor to device
            auto func = nppiRGBToYCbCr420_8u_C3P3R_Ctx;
            if(pformat == PixelFormat::I420){
                func = nppiRGBToYUV420_8u_C3P3R_Ctx;
            }

            auto status = func(pSrc, srcStep, pDst, dstStep, NppiSize{width, height}, streamCtx);
            HMP_REQUIRE(status == NPP_SUCCESS, 
                "nppiRGBToXXX420_8u_P3C3R_Ctx failed with status={}", status);
        }
    }
    else{
        HMP_REQUIRE(false, "unsupport pixel format", pformat);
    }

    return dst;
}



TensorList &yuv_resize_cuda(TensorList &dst, const TensorList &src,
                          PixelFormat format, ImageFilterMode mode)
{
    auto streamCtx = makeNppStreamContext();
    auto batch = dst[0].size(0);
    auto nppMode = getNPPIFilterMode(mode);

    HMP_DISPATCH_ALL_TYPES(src[0].scalar_type(), "yuv_resize_cuda", [&](){
        for(size_t i = 0; i < src.size(); ++i){
            for(int64_t j = 0; j < batch; ++j){
                auto s = src[i].select(0, j);
                auto d = dst[i].select(0, j);
                nppiResize<scalar_t>(d, s, nppMode, streamCtx);
            }
        }
    });

    return dst;
}


TensorList &yuv_rotate_cuda(TensorList &dst, const TensorList &src,
                           PixelFormat format, ImageRotationMode rotate)
{
    auto streamCtx = makeNppStreamContext();
    auto batch = dst[0].size(0);

    HMP_DISPATCH_ALL_TYPES(src[0].scalar_type(), "yuv_rotate_cuda", [&](){
        for(size_t i = 0; i < src.size(); ++i){
            for(int64_t j = 0; j < batch; ++j){
                auto s = src[i].select(0, j);
                auto d = dst[i].select(0, j);
                nppiRotate<scalar_t>(d, s, rotate, streamCtx);
            }
        }
    });

    return dst;
}
    
TensorList &yuv_mirror_cuda(TensorList &dst, const TensorList &src,
                           PixelFormat format, ImageAxis axis)
{
    auto streamCtx = makeNppStreamContext();
    auto batch = dst[0].size(0);

    HMP_DISPATCH_ALL_TYPES(src[0].scalar_type(), "yuv_mirror_cuda", [&](){
        for(size_t i = 0; i < src.size(); ++i){
            for(int64_t j = 0; j < batch; ++j){
                auto s = src[i].select(0, j);
                auto d = dst[i].select(0, j);
                nppiMirror<scalar_t>(d, s, axis, streamCtx);
            }
        }
    });

    return dst;
}

Tensor &img_resize_cuda(Tensor &dst, const Tensor &src, ImageFilterMode mode_)
{
    auto streamCtx = makeNppStreamContext();
    auto batch = src.size(0);
    auto mode = getNPPIFilterMode(mode_);
    HMP_DISPATCH_ALL_TYPES(src.scalar_type(), "img_resize_cuda", [&](){
        for(int64_t i = 0; i < batch; ++i){
            auto d = dst.select(0, i);
            auto s = src.select(0, i);
            nppiResize<scalar_t>(d, s, mode, streamCtx);
        }
    });

    return dst;
}



Tensor &img_rotate_cuda(Tensor &dst, const Tensor &src, ImageRotationMode mode, ChannelFormat cformat)
{
    HMP_REQUIRE(cformat == ChannelFormat::NCHW,
        "img_rotate_cuda: only support NCHW layout");

    auto streamCtx = makeNppStreamContext();
    auto batch = src.size(0);
    HMP_DISPATCH_ALL_TYPES(src.scalar_type(), "img_rotate_cuda", [&](){
        for(int64_t i = 0; i < batch; ++i){
            auto d = dst.select(0, i);
            auto s = src.select(0, i);
            nppiRotate<scalar_t>(d, s, mode, streamCtx);
        }
    });

    return dst;
}

Tensor &img_mirror_cuda(Tensor &dst, const Tensor &src, ImageAxis axis, ChannelFormat cformat)
{
    HMP_REQUIRE(cformat == ChannelFormat::NCHW,
        "img_mirror_cpu: only support NCHW layout");

    auto streamCtx = makeNppStreamContext();
    auto batch = src.size(0);
    HMP_DISPATCH_ALL_TYPES(src.scalar_type(), "img_rotate_cuda", [&](){
        for(int64_t i = 0; i < batch; ++i){
            auto d = dst.select(0, i);
            auto s = src.select(0, i);
            nppiMirror<scalar_t>(d, s, axis, streamCtx);
        }
    });

    return dst;
}


//HMP_DEVICE_DISPATCH(kCUDA, yuv_to_rgbx_stub, &yuv_to_rgbx_cuda)
//HMP_DEVICE_DISPATCH(kCUDA, rgb_to_yuv_stub, &rgb_to_yuv_cuda)
//HMP_DEVICE_DISPATCH(kCUDA, yuv_resize_stub, &yuv_resize_cuda)
//HMP_DEVICE_DISPATCH(kCUDA, yuv_rotate_stub, &yuv_rotate_cuda)
//HMP_DEVICE_DISPATCH(kCUDA, yuv_mirror_stub, &yuv_mirror_cuda)
//HMP_DEVICE_DISPATCH(kCUDA, img_resize_stub, &img_resize_cuda)
//HMP_DEVICE_DISPATCH(kCUDA, img_rotate_stub, &img_rotate_cuda)
//HMP_DEVICE_DISPATCH(kCUDA, img_mirror_stub, &img_mirror_cuda)

}}} //namespace hmp::kernel