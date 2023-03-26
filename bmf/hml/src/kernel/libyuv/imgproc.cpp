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

#include <kernel/imgproc.h>
#include <kernel/kernel_utils.h>
#include <kernel/cpu/libyuv_wrapper.h>

namespace hmp{
namespace kernel{
namespace{


Tensor &yuv_to_rgb_cpu(Tensor &dst, const TensorList &src, PPixelFormat pformat, ChannelFormat cformat)
{
    auto out = dst;
    if(cformat == ChannelFormat::NCHW){
        out = empty({dst.size(0), dst.size(2), dst.size(3), dst.size(1)}, dst.options());
    }

    auto width = out.size(2);
    auto height = out.size(1);

    for(int64_t i = 0; i < out.size(0); ++i){
        auto d = out.select(0, i);
        TensorList ss;
        for(auto &s : src){
            ss.push_back(s.select(0, i));
        }

        switch (pformat)
        {
        case PPixelFormat::I420:
            HMP_REQUIRE(ss.size() == 3, "yuv_to_rgb_cpu: expect 3 input planes, got {}", ss.size());
            libyuv::I420ToRAW( //I420ToRGB24 have invalid order??
                ss[0].data<uint8_t>(), ss[0].stride(0), 
                ss[1].data<uint8_t>(), ss[1].stride(0),
                ss[2].data<uint8_t>(), ss[2].stride(0),
                d.data<uint8_t>(), d.stride(0), d.size(1), d.size(0));
            break;
        case PPixelFormat::H420:
            HMP_REQUIRE(ss.size() == 3, "yuv_to_rgb_cpu: expect 3 input planes, got {}", ss.size());
            libyuv::H420ToRAW( //H420ToRGB24 have invalid order??
                ss[0].data<uint8_t>(), ss[0].stride(0), 
                ss[1].data<uint8_t>(), ss[1].stride(0),
                ss[2].data<uint8_t>(), ss[2].stride(0),
                d.data<uint8_t>(), d.stride(0), d.size(1), d.size(0));
            break;

        default:
            HMP_REQUIRE(false,
                "yuv_to_rgb_cpu: unsupport pixel format {}", pformat);
            break;
        }
    }

    if(cformat == ChannelFormat::NCHW){
        copy(dst, out.permute({0, 3, 1, 2}));
    }

    return dst;
}



TensorList &rgb_to_yuv_cpu(TensorList &dst, const Tensor &src, PPixelFormat pformat, ChannelFormat cformat)
{
    auto w = cformat == ChannelFormat::NHWC ? src : src.permute({0, 2, 3, 1}).contiguous();
    auto width = w.size(2);
    auto height = w.size(1);

    for(int64_t i = 0; i < src.size(0); ++i){
        auto s = w.select(0, i);
        TensorList ds;
        for(auto &d : dst){
            ds.push_back(d.select(0, i));
        }

        switch (pformat)
        {
        case PPixelFormat::I420:
            HMP_REQUIRE(ds.size() == 3, "rgb_to_yuv_cpu: expect 3 input planes, got {}", ds.size());
            libyuv::RAWToI420( 
                s.data<uint8_t>(), s.stride(0),
                ds[0].data<uint8_t>(), ds[0].stride(0), 
                ds[1].data<uint8_t>(), ds[1].stride(0),
                ds[2].data<uint8_t>(), ds[2].stride(0), 
                s.size(1), s.size(0));
            break;
        case PPixelFormat::H420:
            HMP_REQUIRE(ds.size() == 3, "rgb_to_yuv_cpu: expect 3 input planes, got {}", ds.size());
            libyuv::RAWToJ420(  //FIXME: No RAWToJ420 found 
                s.data<uint8_t>(), s.stride(0),
                ds[0].data<uint8_t>(), ds[0].stride(0), 
                ds[1].data<uint8_t>(), ds[1].stride(0),
                ds[2].data<uint8_t>(), ds[2].stride(0), 
                s.size(1), s.size(0));
            break;

        default:
            HMP_REQUIRE(false,
                "rgb_to_yuv_cpu: unsupport pixel format {}", pformat);
            break;
        }
    }

    return dst;
}



TensorList &yuv_resize_cpu(TensorList &dst, const TensorList &src,
                          PPixelFormat format, ImageFilterMode mode)
{
    auto batch = src[0].size(0);
    auto m = getLibYUVFilterMode(mode);

    HMP_DISPATCH_UNSIGNED_INTEGRAL_TYPES(src[0].scalar_type(), "yuv_resize_cpu", [&]() {
        for (int64_t i = 0; i < src.size(); ++i){
            for (int64_t j = 0; j < batch; ++j){
                auto d = dst[i].select(0, j);
                auto s = src[i].select(0, j);
                libyuvScalePlane<scalar_t>(d, s, m);
            }
        }
    });

    return dst;
}

TensorList &yuv_rotate_cpu(TensorList &dst, const TensorList &src,
                           PPixelFormat format, ImageRotationMode rotate)
{
    auto batch = src[0].size(0);
    auto m = getLibYUVRotationMode(rotate);

    HMP_DISPATCH_UNSIGNED_INTEGRAL_TYPES(src[0].scalar_type(), "yuv_rotate_cpu", [&]() {
        for (int64_t i = 0; i < src.size(); ++i){
            for (int64_t j = 0; j < batch; ++j){
                auto d = dst[i].select(0, j);
                auto s = src[i].select(0, j);
                libyuvRotatePlane<scalar_t>(d, s, m);
            }
        }
    });

    return dst;
}


TensorList &yuv_mirror_cpu(TensorList &dst, const TensorList &src, PPixelFormat format, ImageAxis axis)
{
    auto batch = src[0].size(0);
    HMP_DISPATCH_UNSIGNED_INTEGRAL_TYPES(src[0].scalar_type(), "yuv_mirror_cpu", [&]() {
        for (int64_t i = 0; i < src.size(); ++i){
            for (int64_t j = 0; j < batch; ++j){
                auto d = dst[i].select(0, j);
                auto s = src[i].select(0, j);
                libyuvMirrorPlane<scalar_t>(d, s, axis);
            }
        }
    });

    return dst;
}

Tensor &img_resize_cpu(Tensor &dst, const Tensor &src, ImageFilterMode mode_, ChannelFormat cformat)
{
    HMP_REQUIRE(cformat == ChannelFormat::NCHW,
        "img_resize_cpu: only support NCHW layout");

    auto batch = src.size(0);
    auto mode = getLibYUVFilterMode(mode_);
    HMP_DISPATCH_UNSIGNED_INTEGRAL_TYPES(src.scalar_type(), "img_resize_cpu", [&](){
        for(int64_t i = 0; i < batch; ++i){
            auto d = dst.select(0, i);
            auto s = src.select(0, i);
            libyuvScalePlane<scalar_t>(d, s, mode);
        }
    });

    return dst;
}



Tensor &img_rotate_cpu(Tensor &dst, const Tensor &src, ImageRotationMode mode_, ChannelFormat cformat)
{
    HMP_REQUIRE(cformat == ChannelFormat::NCHW,
        "img_rotate_cpu: only support NCHW layout");

    auto batch = src.size(0);
    auto mode = getLibYUVRotationMode(mode_);
    HMP_DISPATCH_UNSIGNED_INTEGRAL_TYPES(src.scalar_type(), "img_rotate_cpu", [&](){
        for(int64_t i = 0; i < batch; ++i){
            auto d = dst.select(0, i);
            auto s = src.select(0, i);
            libyuvRotatePlane<scalar_t>(d, s, mode);
        }
    });

    return dst;
}


Tensor &img_mirror_cpu(Tensor &dst, const Tensor &src, ImageAxis axis, ChannelFormat cformat)
{
    HMP_REQUIRE(cformat == ChannelFormat::NCHW,
        "img_mirror_cpu: only support NCHW layout");

    auto batch = src.size(0);
    HMP_DISPATCH_UNSIGNED_INTEGRAL_TYPES(src.scalar_type(), "img_mirror_cpu", [&](){
        for(int64_t i = 0; i < batch; ++i){
            auto d = dst.select(0, i);
            auto s = src.select(0, i);
            libyuvMirrorPlane<scalar_t>(d, s, axis);
        }
    });

    return dst;
}


//HMP_DEVICE_DISPATCH(kCPU, yuv_to_rgb_stub, &yuv_to_rgb_cpu)
//HMP_DEVICE_DISPATCH(kCPU, rgb_to_yuv_stub, &rgb_to_yuv_cpu)
//HMP_DEVICE_DISPATCH(kCPU, yuv_resize_stub, &yuv_resize_cpu)
//HMP_DEVICE_DISPATCH(kCPU, yuv_rotate_stub, &yuv_rotate_cpu)
//HMP_DEVICE_DISPATCH(kCPU, yuv_mirror_stub, &yuv_mirror_cpu)
//HMP_DEVICE_DISPATCH(kCPU, img_resize_stub, &img_resize_cpu)
//HMP_DEVICE_DISPATCH(kCPU, img_rotate_stub, &img_rotate_cpu)
//HMP_DEVICE_DISPATCH(kCPU, img_mirror_stub, &img_mirror_cpu)


}}} //namespace hmp::kernel