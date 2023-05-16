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

Frame::Frame(const Tensor &data, const PixelInfo &pix_info)
    : Frame({data}, data.size(1), data.size(0), pix_info)
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

Frame Frame::reformat(const PixelInfo &pix_info)
{
    if (pix_info_.format() == PF_RGB24) {
        auto yuv = img::rgb_to_yuv(data_[0], pix_info, kNHWC);
        return Frame(yuv, width_, height_, pix_info);

    } else if (pix_info.format() == PF_RGB24) {
        auto rgb = img::yuv_to_rgb(data_, pix_info_, kNHWC);
        return Frame({rgb}, width_, height_, pix_info);
    }

    HMP_REQUIRE(false, "{} to {} not support", stringfy(pix_info_.format()), stringfy(pix_info.format()));
}

std::string stringfy(const Frame &frame)
{
    return fmt::format("Frame({}, {}, {}, ({}, {}, {}))",
        frame.device(), frame.dtype(), frame.format(),
        frame.nplanes(), frame.height(), frame.width());
}

} //namespace hmp
