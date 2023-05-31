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

#include <hmp/imgproc.h>
#include <kernel/image_iter.h>

namespace hmp{
namespace kernel{

template<typename T, PPixelFormat format, ChannelFormat cformat>
struct YUV2RGB
{
    RGBIter<T, cformat> rgb_iter;
    YUVIter<T, format> yuv_iter;
    using wtype = Vector<float, 3>;
    using otype = Vector<T, 3>;
    using cast_type = Vector<uint8_t, 3>; //FIXME

    YUV2RGB(Tensor& rgb, const TensorList &yuv)
        : rgb_iter(rgb), yuv_iter(yuv)
    {
        HMP_REQUIRE(rgb_iter.width() == yuv_iter.width() && rgb_iter.height() == yuv_iter.height(),
            "YUV2RGB: yuv and rgb image size are not matched, yuv:{}, rgb:{}",
            SizeArray{yuv_iter.width(), yuv_iter.height()}, 
            SizeArray{rgb_iter.width(), rgb_iter.height()});
    }

    HMP_HOST_DEVICE inline void operator()(int batch, int w, int h)
    {
        wtype yuv = yuv_iter.get(batch, w, h);
        wtype rgb(0, 0, 0);

        //FIXME: only support 8bit pixel data 

        if(format == PPixelFormat::H420 ||
           format == PPixelFormat::H422 || 
           format == PPixelFormat::H444){ //BT.709 limited range
            yuv -= wtype(16.f, 128.f, 128.f); //
            rgb[0] = yuv.dot(wtype{1.164384f, 0.f, 1.792741f});
            rgb[1] = yuv.dot(wtype{1.164384f, -0.213249f, -0.532909f});
            rgb[2] = yuv.dot(wtype{1.164384f, 2.112402f, 0.f});
        }
        else if(format == PPixelFormat::I420 || 
                format == PPixelFormat::I422 || 
                format == PPixelFormat::I444 ||
                format == PPixelFormat::NV21 ||
                format == PPixelFormat::NV12
                ){ //BT.601 limited range
            yuv -= wtype(16.f, 128.f, 128.f); //
            rgb[0] = yuv.dot(wtype{1.164384f, 0.f, 1.596027f});
            rgb[1] = yuv.dot(wtype{1.164384f, -0.391762f, -0.812968f});
            rgb[2] = yuv.dot(wtype{1.164384f, 2.017232f, 0.f});
        }
        else if(format == PPixelFormat::P010 ||
                format == PPixelFormat::U420 ||
                format == PPixelFormat::U422 ||
                format == PPixelFormat::U444
                ){ //BT.2020 10 bit limited range
            // yuv -= wtype(64.f, 512.f, 512.f); //
            yuv -= wtype((float)(64 << 6), (float)(512 << 6), (float)(512 << 6));
            rgb[0] = yuv.dot(wtype{1.168932f, 0.f, 1.723707f});
            rgb[1] = yuv.dot(wtype{1.168932f, -0.192351, -0.667873f});
            rgb[2] = yuv.dot(wtype{1.168932f, 2.199229f, 0.f});
        }
        else{
            //zeros
        }

        auto rgb_out = saturate_cast<otype>(rgb);
        rgb_iter.set(batch, w, h, rgb_out);
    }
};


template<typename T, PPixelFormat format, ChannelFormat cformat>
struct RGB2YUV
{
    RGBIter<T, cformat> rgb_iter;
    YUVIter<T, format> yuv_iter;
    using wtype = Vector<float, 3>;
    using otype = Vector<T, 3>;
    using cast_type = Vector<uint8_t, 3>; //FIXME

    RGB2YUV(TensorList& yuv, const Tensor &rgb)
        : rgb_iter(rgb), yuv_iter(yuv)
    {
        HMP_REQUIRE(rgb_iter.width() == yuv_iter.width() && rgb_iter.height() == yuv_iter.height(),
            "RGB2YUV: yuv and rgb image size are not matched, yuv:{}, rgb:{}",
            SizeArray{yuv_iter.width(), yuv_iter.height()}, 
            SizeArray{rgb_iter.width(), rgb_iter.height()});
    }

    HMP_HOST_DEVICE inline void operator()(int batch, int w, int h)
    {
        wtype rgb = rgb_iter.get(batch, w, h);
        wtype yuv(0, 0, 0);

        if(format == PPixelFormat::H420 ||
           format == PPixelFormat::H422 || 
           format == PPixelFormat::H444){ //BT.709 limited range
            yuv[0] = rgb.dot(wtype{0.18258588f,  0.61423059f,  0.06200706f});
            yuv[1] = rgb.dot(wtype{-0.10064373f, -0.33857195f,  0.43921569f});
            yuv[2] = rgb.dot(wtype{0.43921569f, -0.39894216f, -0.04027352f});
            yuv += wtype(16.f, 128.f, 128.f);
        }
        else if(format == PPixelFormat::I420 || 
                format == PPixelFormat::I422 || 
                format == PPixelFormat::I444 ||
                format == PPixelFormat::NV21 ||
                format == PPixelFormat::NV12
                ){ //BT.601
            yuv[0] = rgb.dot(wtype{0.25678824f,  0.50412941f,  0.09790588f});
            yuv[1] = rgb.dot(wtype{-0.1482229f,  -0.29099279f,  0.43921569f});
            yuv[2] = rgb.dot(wtype{0.43921569f, -0.36778831f, -0.07142737f});
            yuv += wtype(16.f, 128.f, 128.f);
        }
        else if(format == PPixelFormat::P010 ||
                format == PPixelFormat::U420 ||
                format == PPixelFormat::U422 ||
                format == PPixelFormat::U444
                ){ //BT.2020 10 bit limited range
            // yuv -= wtype(64.f, 512.f, 512.f); //
            yuv += wtype((float)(64 << 6), (float)(512 << 6), (float)(512 << 6));
            yuv[0] = rgb.dot(wtype{0.22473507f, 0.58001666f, 0.05073007f});
            yuv[1] = rgb.dot(wtype{-0.11945098f, -0.30828992f, 0.42774090f});
            yuv[2] = rgb.dot(wtype{0.42774090f, -0.39333830f, -0.03440260f});
        }
        else{
            //zeros
        }

        //yuv[0] = clamp<float>(yuv[0], 16, 235);
        //yuv[1] = clamp<float>(yuv[1], 16, 240);
        //yuv[2] = clamp<float>(yuv[2], 16, 240);

        auto yuv_out = saturate_cast<otype>(yuv);
        yuv_iter.set(batch, w, h, yuv_out);
    }
};

template<typename T, PPixelFormat dformat, PPixelFormat sformat>
struct YUV2YUV
{
    // RGBIter<T, cformat> rgb_iter;
    YUVIter<T, sformat> src_iter;
    YUVIter<T, dformat> dst_iter;
    using wtype = Vector<float, 3>;
    using otype = Vector<T, 3>;
    using cast_type = Vector<uint8_t, 3>; //FIXME

    YUV2YUV(TensorList& dst, const TensorList &src)
        : src_iter(src), dst_iter(dst)
    {
        HMP_REQUIRE(src_iter.width() == dst_iter.width() && src_iter.height() == dst_iter.height(),
            "YUV2YUV: yuv and rgb image size are not matched, dst:{}, src:{}",
            SizeArray{dst_iter.width(), dst_iter.height()}, 
            SizeArray{src_iter.width(), src_iter.height()});
    }

    HMP_HOST_DEVICE inline void operator()(int batch, int w, int h)
    {
        wtype src = src_iter.get(batch, w, h);
        wtype dst = src;

        auto yuv_out = saturate_cast<cast_type>(dst);
        dst_iter.set(batch, w, h, yuv_out);
    }
};

}} //namespace hmp::kernel