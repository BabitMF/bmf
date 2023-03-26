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
#include <hmp/imgproc/image.h>
#include <hmp/imgproc.h>
#include <hmp/format.h>

namespace hmp{


Frame::Frame(const TensorList &data, int width, int height, const PixelInfo& pix_info)
    : pix_info_(pix_info), width_(width), height_(height)
{
    pix_desc_ = PixelFormatDesc(pix_info_.format());

    // convert to HWC layout if pixel format is supported
    if(pix_desc_.defined()){
        data_ = img::frame_format(data, pix_desc_, width, height);
    }
    else{
        data_ = data;
    }
}

Frame::Frame(const TensorList &data, const PixelInfo &pix_info)
    : Frame(data, data[0].size(1), data[0].size(0), pix_info)
{
}


Frame::Frame(int width, int height, const PixelInfo &pix_info, const Device &device)
    : pix_info_(pix_info), width_(width), height_(height)
{
    pix_desc_ = PixelFormatDesc(pix_info_.format());

    HMP_REQUIRE(pix_desc_.defined(), 
        "PixelFormat {} is not supported by hmp", pix_info_.format());

    auto options = TensorOptions(device).dtype(pix_desc_.dtype());
    for(int i = 0; i < pix_desc_.nplanes(); ++i){
        SizeArray shape{pix_desc_.infer_height(height, i),
                        pix_desc_.infer_width(width, i),
                        pix_desc_.channels(i)};

        data_.push_back(empty(shape, options));
    }
}


Frame Frame::to(const Device &device, bool non_blocking) const
{
    TensorList out;
    for(auto &d : data_){
        out.push_back(d.to(device, non_blocking));
    }
    return Frame(out, width_, height_, pix_info_);
}


Frame Frame::to(DeviceType device, bool non_blocking) const
{
    TensorList out;
    for(auto &d : data_){
        out.push_back(d.to(device, non_blocking));
    }
    return Frame(out, width_, height_, pix_info_);
}



Frame &Frame::copy_(const Frame &from)
{
    HMP_REQUIRE(format() == from.format(), 
        "Can't copy from different PixelFormat {}, expect {}",
        from.format(), format());
    for(size_t i = 0; i < data_.size(); ++i){
        data_[i].copy_(from.data_[i]);
    }
    return *this;
}

Frame Frame::clone() const
{
    TensorList out;
    for(auto &d : data_){
        out.push_back(d.clone());
    }
    return Frame(out, width_, height_, pix_info_);
}


Frame Frame::crop(int left, int top, int w, int h) const
{
    HMP_REQUIRE(pix_desc_.defined(), 
        "Frame::crop: pixel format {} is not supported", pix_desc_.format());

    auto width = this->width();
    auto height = this->height();
    left = wrap_size(left, width);
    top = wrap_size(top, height);

    auto right = left + w;
    auto bottom = top + h;
    HMP_REQUIRE(left < right && right <= width, 
        "Frame::crop expect left({}) < right({}) and right <= {}", left, right, width);
    HMP_REQUIRE(top < bottom && bottom <= height, 
        "Frame::crop expect top({}) < bottom({}) and bottom <= {}", top, bottom, height);

    //normalize
    double left_d = double(left)/width;
    double right_d = double(right)/width;
    double bottom_d = double(bottom)/height;
    double top_d = double(top)/height;

    //
    TensorList out;
    for(auto &d : data_){
        auto l = int(std::round(left_d*d.size(1)));
        auto r = int(std::round(right_d*d.size(1)));
        auto b = int(std::round(bottom_d*d.size(0)));
        auto t = int(std::round(top_d*d.size(0)));

        out.push_back(d.slice(0, t, b).slice(1, l, r));
    }

    return Frame(out, w, h, pix_info_);
}


Image Frame::to_image(ChannelFormat cformat) const
{
    if(pix_info_.is_rgbx()){
        HMP_REQUIRE(data_.size() == 1, "Internal error");
        return Image(data_[0], kNHWC, pix_info_.color_model()).to(cformat);
    }
    else{
#ifndef HMP_ENABLE_MOBILE
        HMP_REQUIRE(pix_desc_.defined(),
                    "Frame::to_image: pixel format {} is not supported", pix_desc_.format());
        auto data = img::yuv_to_rgb(data_, pix_info_, cformat);
        return Image(data, cformat, pix_info_.color_model());
#else
        HMP_REQUIRE(false,
                    "Frame::to_image: pixel format {} is not supported", pix_desc_.format());
#endif
    }
}


Frame Frame::from_image(const Image &image, const PixelInfo &pix_info)
{
    if(pix_info.is_rgbx()){
        auto pix_desc = PixelFormatDesc(pix_info.format());
        HMP_REQUIRE(pix_desc.defined() && pix_desc.channels(0) == image.nchannels(), 
            "Frame::from_image: expect image has {} channels, got {}",
            pix_desc.channels(), image.nchannels());
        HMP_REQUIRE(image.format() == kNHWC, 
            "Frame::from_image: expect image has NHWC layout");
        HMP_REQUIRE(image.dtype() == pix_desc.dtype(),
            "Frame:from_image: expect image has dtype {}, got {}",
            pix_desc.dtype(), image.dtype());
        return Frame({image.data()}, pix_info);
    }
    else{
#ifndef HMP_ENABLE_MOBILE
        auto yuv = img::rgb_to_yuv(image.data(), pix_info, image.format());
        return Frame(yuv, image.width(), image.height(), pix_info);
#else
        HMP_REQUIRE(false,
                    "Frame::from_image: pixel format {} is not supported", pix_info.format());
#endif //HMP_ENABLE_MOBILE
    }
}



/////////////////////////// Image ////////////////////////////

Image::Image(const Tensor &data, ChannelFormat format, const ColorModel &cm)
    : Image(data, format)
{
    cm_ = cm;
}


Image::Image(const Tensor &data, ChannelFormat format)
    : format_(format)
{
    HMP_REQUIRE(data.dim() == 2 || data.dim() == 3,
        "Image: expect data has 2 or 3 dims, got {}", data.dim());

    //convert to standard layout
    if(data.dim() == 2){
        if(format == kNCHW){
            data_ = data.unsqueeze(0);
        }
        else{
            data_ = data.unsqueeze(-1);
        }
    }
    else{
        data_ = data.alias();
    }
}


Image::Image(int width, int height, int channels, ChannelFormat format, const TensorOptions &options)
{
    if(format == kNCHW){
        data_ = empty({channels, height, width}, options);
    }
    else{
        data_ = empty({height, width, channels}, options);
    }
    format_ = format;
}


Image Image::to(const Device &device, bool non_blocking) const
{
    auto data = data_.to(device, non_blocking);
    return Image(data, format_, cm_);
}

Image Image::to(DeviceType device, bool non_blocking) const
{
    auto data = data_.to(device, non_blocking);
    return Image(data, format_, cm_);
}

Image Image::to(ScalarType dtype) const
{
    auto data = data_.to(dtype);
    return Image(data, format_, cm_);
}

Image Image::to(ChannelFormat format, bool contiguous) const
{
    Tensor data = data_;
    if(format == ChannelFormat::NCHW && format_ == ChannelFormat::NHWC){
        data = data_.permute({2, 0, 1});
    }
    else if(format == ChannelFormat::NHWC && format_ == ChannelFormat::NCHW){
        data = data_.permute({1, 2, 0});
    }

    if(contiguous){
        data = data.contiguous();
    }
    return Image(data, format, cm_);
}


Image &Image::copy_(const Image &from)
{
    HMP_REQUIRE(from.format() == format(), 
        "Image::copy_: expect channel format {}, got {}", format(), from.format());

    data_.copy_(from.data_);
    return *this;
}

Image Image::clone() const
{
    return Image(data_.clone(), format_, cm_);
}


Image Image::crop(int left, int top, int w, int h) const
{
    auto width = this->width();
    auto height = this->height();
    left = wrap_size(left, width);
    top = wrap_size(top, height);

    auto right = left + w;
    auto bottom = top + h;
    HMP_REQUIRE(left < right && right <= width, 
        "Image::crop expect left({}) < right({}) and right <= {}", left, right, width);
    HMP_REQUIRE(top < bottom && bottom <= height, 
        "Image::crop expect top({}) < bottom({}) and bottom <= {}", top, bottom, height);

    auto data = data_.slice(wdim(), left, right).slice(hdim(), top, bottom);
    return Image(data, format_, cm_);
}


Image Image::select(int channel) const
{
    auto data = data_.slice(cdim(), channel, channel+1);
    return Image(data, format_, cm_); 
}


std::string stringfy(const Frame &frame)
{
    return fmt::format("Frame({}, {}, {}, ({}, {}, {}))",
        frame.device(), frame.dtype(), frame.format(),
        frame.nplanes(), frame.height(), frame.width());
}

std::string stringfy(const Image &image)
{
    return fmt::format("Image({}, {}, {})",
        image.device(), image.dtype(), image.format(),
        image.data().shape());
}


} //namespace hmp