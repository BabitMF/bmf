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

#include <hmp/tensor.h>
#include <kernel/vector.h>
#include <kernel/kernel_utils.h>

namespace hmp{
namespace kernel{


template<typename T>
struct ImageIter
{
    using value_type = T;
    using index_type = int;

    ImageIter() = default;

    HMP_HOST_DEVICE ImageIter(
        value_type *ptr, int width, 
        int height, int row_stride)
        : ptr_(ptr), width_(width), height_(height),
         row_stride_(row_stride)
    {
    }

    HMP_HOST_DEVICE inline value_type& operator()(int x, int y)
    {
        auto idx = index_type(y)*row_stride_ + x;
        return ptr_[idx];
    }

    HMP_HOST_DEVICE inline const value_type& operator()(int x, int y) const
    {
        return ptr_[y*row_stride_ + x];
    }

    value_type *ptr_ = 0;
    int row_stride_ = 0; 
    int width_ = 0, height_ = 0;
};


enum class ImageBorderType
{
    Replicate,
    Constant //RGB - zeros, alpha - one
};
const static ImageBorderType kBReplicate = ImageBorderType::Replicate;
const static ImageBorderType kBConstant = ImageBorderType::Constant;


template<typename Pixel>
struct DefaultPixelValue
{
    HMP_HOST_DEVICE static Pixel value() {
        return Pixel();
    } 
};


template<typename T>
struct DefaultPixelValue<Vector<T, 4>>
{
    HMP_HOST_DEVICE static Vector<T, 4> value() {
        return Vector<T, 4>(0, 0, 0, 
            std::is_integral<T>::value ? std::numeric_limits<T>::max() : T(1));
    } 
};


template<typename T>
struct ImageIndexer
{
    using index_type = T;

    ImageIndexer() = default;

    ImageIndexer(int batch, int width, int height, int batch_stride, int row_stride)
        : batch_(batch), width_(width), height_(height), batch_stride_(batch_stride), row_stride_(row_stride)
    {
    }

    HMP_HOST_DEVICE inline index_type index(int batch, int x, int y) const
    {
        return index_type(batch)*batch_stride_ + index_type(y) * row_stride_ + x;
    }

    HMP_HOST_DEVICE int batch() const
    {
        return batch_;
    }

    HMP_HOST_DEVICE int width() const
    {
        return width_;
    }

    HMP_HOST_DEVICE int height() const
    {
        return height_;
    }

    int batch_stride_ = 0, row_stride_ = 0; 
    int batch_ = 0, width_ = 0, height_ = 0;
};


template<typename Pixel, ChannelFormat Format = kNHWC>
struct ImageSeqIter;

template<typename Pixel>
struct ImageSeqIter<Pixel, kNHWC> : public ImageIndexer<int>
{
    using value_type = Pixel;

    ImageSeqIter() = default;

    ImageSeqIter(const Tensor &t, ChannelFormat format = kNHWC, ImageBorderType border = kBReplicate)
    {
        HMP_REQUIRE(t.stride(-1) == 1, "ImageSeqIter require last dim stride is 1, got {}", t.stride(-1));

        border_ = border;
        if (format == kNCHW)
        {
            HMP_REQUIRE(t.dim() == 4 || t.dim() == 3,
                        "ImageSeqIter require 3 or 4 dims, got {}", t.dim());
            HMP_REQUIRE(Pixel::size() == 1, "ImageSeqIter invalid Vector type, expect size=1, got {}", Pixel::size());

            int64_t batch;
            if (t.dim() == 4){
                HMP_REQUIRE(t.size(1) * t.stride(1) == t.stride(0),
                            "ImageSeqIter require batch dim contiguous");
                batch = t.size(0) * t.size(1);
            }
            else{
                batch = t.size(0);
            }

            ptr_ = (Pixel*)(t.data<typename Pixel::value_type>());
            width_ = t.size(-1);
            height_ = t.size(-2);
            batch_ = batch;
            batch_stride_ = t.stride(-3) / Pixel::size();
            row_stride_ = t.stride(-2) / Pixel::size();
        }
        else
        {
            HMP_REQUIRE(t.dim() == 4,
                        "ImageSeqIter require 4 dims, got {}", t.dim());
            HMP_REQUIRE(t.stride(2) == t.size(3),
                        "ImageSeqIter require Vec(-2) dim stride is contiguous, expect {}, got {}", t.size(3), t.stride(2));
            HMP_REQUIRE(Pixel::size() == t.size(3),
                        "ImageSeqIter invalid Vector type, expect size={}, got {}", t.size(-1), Pixel::size());

            ptr_ = (Pixel*)(t.data<typename Pixel::value_type>());
            width_ = t.size(2);
            height_ = t.size(1);
            batch_ = t.size(0);
            batch_stride_ = t.stride(0) / Pixel::size();
            row_stride_ = t.stride(1) / Pixel::size();
        }
    }


    HMP_HOST_DEVICE inline Pixel get(int batch, int x, int y) const
    {
        if(border_ == kBReplicate){
            x = clamp(x, 0, width_ - 1);
            y = clamp(y, 0, height_ - 1);
            return ptr_[index(batch, x, y)];
        }
        else{
            if(x < 0 || x >= width_ || y < 0 || y >= height_){
                return DefaultPixelValue<Pixel>::value();
            }
            else{
                return ptr_[index(batch, x, y)];
            }
        }
    }

    HMP_HOST_DEVICE inline void set(int batch, int x, int y, const Pixel &pix) const
    {
        if(border_ == kBReplicate){
            x = clamp(x, 0, width_ - 1);
            y = clamp(y, 0, height_ - 1);
        }
        else if(x < 0 || x >= width_ || y < 0 || y >= height_){
            return;
        }
        ptr_[index(batch, x, y)] = pix;
    }

    static ImageSeqIter from_tensor(const Tensor &t, ChannelFormat format = kNCHW, ImageBorderType border = kBReplicate)
    {
        return ImageSeqIter(t, format, border);
    }


    ImageBorderType border_ = kBReplicate;
    Pixel *ptr_ = 0;
};



template<typename Pixel>
struct ImageSeqIter<Pixel, kNCHW> : public ImageIndexer<int>
{
    using value_type = Pixel;

    ImageSeqIter() = default;

    ImageSeqIter(const Tensor &t, ChannelFormat format = kNCHW, ImageBorderType border = kBReplicate)
    {
        HMP_REQUIRE(format == kNCHW, "ImageSeqIter only support NCHW layout");
        HMP_REQUIRE(t.stride(-1) == 1, "ImageSeqIter require last dim stride is 1, got {}", t.stride(-1));
        HMP_REQUIRE(t.dim() == 4 || t.dim() == 3,
                        "ImageSeqIter require 3 or 4 dims, got {}", t.dim());

        if (t.dim() == 4)
        {
            HMP_REQUIRE(t.size(1) == Pixel::size(), "ImageSeqIter internal error");
            for(int i = 0; i < Pixel::size(); ++i){
                ptr_[i] = t.select(1, i).data<typename Pixel::value_type>();
            }
        }
        else
        {
            HMP_REQUIRE(Pixel::size() == 1, "ImageSeqIter internal error");
            ptr_[0] = t.data<typename Pixel::value_type>();
        }

        width_ = t.size(-1);
        height_ = t.size(-2);
        batch_ = t.size(0);
        batch_stride_ = t.stride(0);
        row_stride_ = t.stride(-2);

        border_ = border;
    }

    HMP_HOST_DEVICE inline Pixel get(int batch, int x, int y) const
    {
        Pixel pix;
        if(border_ == kBReplicate){
            x = clamp(x, 0, width_ - 1);
            y = clamp(y, 0, height_ - 1);

            auto idx = index(batch, x, y);
            #pragma unroll
            for(int i = 0; i < Pixel::size(); ++i){
                pix[i] = ptr_[i][idx];
            }
        }
        else{
            if(x < 0 || x >= width_ || y < 0 || y >= height_){
                pix = DefaultPixelValue<Pixel>::value();
            }
            else{
                auto idx = index(batch, x, y);
                #pragma unroll
                for(int i = 0; i < Pixel::size(); ++i){
                    pix[i] = ptr_[i][idx];
                }
            }
        }

        return pix;
    }

    HMP_HOST_DEVICE inline void set(int batch, int x, int y, const Pixel &pix) const
    {
        if(border_ == kBReplicate){
            x = clamp(x, 0, width_ - 1);
            y = clamp(y, 0, height_ - 1);
        }
        else if(x < 0 || x >= width_ || y < 0 || y >= height_){
            return;
        }

        auto idx = index(batch, x, y);
        #pragma unroll
        for(int i = 0; i < Pixel::size(); ++i){
            ptr_[i][idx] = pix[i];
        }
    }

    static ImageSeqIter from_tensor(const Tensor &t, ChannelFormat format = kNHWC, ImageBorderType border = kBReplicate)
    {
        return ImageSeqIter(t, format, border);
    }

    ImageBorderType border_ = kBReplicate;
    typename Pixel::value_type *ptr_[Pixel::size()] = {0};
};


template<typename T, ChannelFormat cformat, int C = 3>
using RGBIter = ImageSeqIter<Vector<T, C>, cformat>;


template<typename T, PPixelFormat format, typename=void>
struct YUVIter;


template<typename T, PPixelFormat format>
struct YUVIter<T, format, 
    std::enable_if_t<format==PPixelFormat::H420 ||
                     format==PPixelFormat::H422 ||
                     format==PPixelFormat::H444 ||
                     format==PPixelFormat::I420 ||
                     format==PPixelFormat::I422 ||
                     format==PPixelFormat::I444
                     >>
{
    const static int hshift = format==PPixelFormat::H420 ||
                              format==PPixelFormat::I420;
    const static int wshift = format==PPixelFormat::H420 ||
                              format==PPixelFormat::H422 ||
                              format==PPixelFormat::I420 ||
                              format==PPixelFormat::I422;

    using value_type = Vector<T, 1>;
    using Iter = ImageSeqIter<value_type>;
    Iter yiter, uiter, viter;

    YUVIter(const TensorList &yuv)
        : yiter(Iter::from_tensor(yuv[0], kNHWC)),
          uiter(Iter::from_tensor(yuv[1], kNHWC)),
          viter(Iter::from_tensor(yuv[2], kNHWC))
    {
        auto uv_width = yiter.width_>>wshift;
        auto uv_height = yiter.height_>>hshift;

        HMP_REQUIRE(uiter.width_ == uv_width && uiter.height_ == uv_height,
             "YUVIter: U plane size is not matched PixelFormat {}, expect {}, got {}",
             format, SizeArray{uv_width, uv_height}, SizeArray{uiter.width_, uiter.height_});
        HMP_REQUIRE(viter.width_ == uv_width && viter.height_ == uv_height,
             "YUVIter: V plane size is not matched PixelFormat {}, expect {}, got {}",
             format, SizeArray{uv_width, uv_height}, SizeArray{viter.width_, viter.height_});
    }

    HMP_HOST_DEVICE inline Vector<T, 3> get(int batch, int w, int h)
    {
        T y = yiter.get(batch, w, h)[0],
          u = uiter.get(batch, w>>wshift, h>>hshift)[0],
          v = viter.get(batch, w>>wshift, h>>hshift)[0];

        return Vector<T, 3>(y, u, v);
    } 

    HMP_HOST_DEVICE inline void set(int batch, int w, int h, const Vector<T, 3> &yuv)
    {
        yiter.set(batch, w, h, value_type(yuv[0]));
        uiter.set(batch, w >> wshift, h >> hshift, value_type(yuv[1]));
        viter.set(batch, w >> wshift, h >> hshift, value_type(yuv[2]));
    } 

    HMP_HOST_DEVICE inline int width() const
    {
        return yiter.width();
    }

    HMP_HOST_DEVICE inline int height() const
    {
        return yiter.height();
    }
};


template<typename T, PPixelFormat format>
struct YUVIter<T, format, 
    std::enable_if_t<format==PPixelFormat::NV21 ||
                     format==PPixelFormat::NV12 ||
                     format==PPixelFormat::NV21_BT709 || 
                     format==PPixelFormat::NV12_BT709
                     >>
{
    const static int wshift = 1;
    const static int hshift = 1;

    using YIter = ImageSeqIter<Vector<T, 1>>;
    using UVIter = ImageSeqIter<Vector<T, 2>>; //
    YIter yiter;
    UVIter uviter;

    YUVIter(const TensorList &yuv)
        : yiter(YIter::from_tensor(yuv[0], kNHWC)),
          uviter(UVIter::from_tensor(yuv[1], kNHWC))
    {
        auto uv_width = yiter.width_>>wshift;
        auto uv_height = yiter.height_>>hshift;

        HMP_REQUIRE(uviter.width_ == uv_width && uviter.height_ == uv_height,
             "YUVIter: UV plane size is not matched PixelFormat {}, expect {}, got {}",
             format, SizeArray{uv_width, uv_height}, SizeArray{uviter.width_, uviter.height_});
    }


    HMP_HOST_DEVICE inline Vector<T, 3> get(int batch, int w, int h)
    {
        T y = yiter.get(batch, w, h)[0];
        auto uv = uviter.get(batch, w>>wshift, h>>hshift);

        if(format == PPixelFormat::NV12 || format == PPixelFormat::NV12_BT709) {
            return Vector<T, 3>(y, uv[0], uv[1]);
        }
        else{
            return Vector<T, 3>(y, uv[1], uv[0]);
        }
    } 

    HMP_HOST_DEVICE inline void set(int batch, int w, int h, const Vector<T, 3> &yuv)
    {
        yiter.set(batch, w, h, typename YIter::value_type(yuv[0]));

        if(format == PPixelFormat::NV12 || format == PPixelFormat::NV12_BT709) {
            uviter.set(batch, w>>wshift, h>>hshift,
                typename UVIter::value_type(yuv[1], yuv[2]));
        }
        else{
            uviter.set(batch, w>>wshift, h>>hshift,
                typename UVIter::value_type(yuv[2], yuv[1]));
        }
    } 

    HMP_HOST_DEVICE inline int width() const
    {
        return yiter.width();
    }

    HMP_HOST_DEVICE inline int height() const
    {
        return yiter.height();
    }
};



#define HMP_IMAGE_CHANNEL_DISPATCH_CASE(Channel, ...) \
    case (Channel):{ \
        using C = Vector<int, Channel>; \
        return __VA_ARGS__();\
    }


#define HMP_DISPATCH_IMAGE_CHANNEL(channel, name, ...) [&](){ \
    switch(channel){ \
        HMP_IMAGE_CHANNEL_DISPATCH_CASE(1, __VA_ARGS__)   \
        HMP_IMAGE_CHANNEL_DISPATCH_CASE(3, __VA_ARGS__)   \
        HMP_IMAGE_CHANNEL_DISPATCH_CASE(4, __VA_ARGS__)   \
        default: \
            HMP_REQUIRE(false, "Unsupported image channels {} in {}, expect 1, 3, 4", channel, #name); \
    } \
}()


#define HMP_CHANNEL_FORMAT_DISPATCH_CASE(Format, ...) \
    case (Format):{ \
        const auto FMT = Format; \
        return __VA_ARGS__();\
    }


#define HMP_DISPATCH_CHANNEL_FORMAT(format, name, ...) [&](){ \
    switch(format){ \
        HMP_CHANNEL_FORMAT_DISPATCH_CASE(kNCHW, __VA_ARGS__)   \
        HMP_CHANNEL_FORMAT_DISPATCH_CASE(kNHWC, __VA_ARGS__)   \
        default: \
            HMP_REQUIRE(false, "Unsupported image channel format {} in {}, expect NCHW, NHWC", format, #name); \
    } \
}()


}} //hmp::kernel
