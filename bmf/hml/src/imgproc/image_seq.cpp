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
#include <hmp/imgproc/image_seq.h>
#include <hmp/imgproc.h>
#include <hmp/format.h>

namespace hmp{

/////////////////////////// FrameSeq /////////////////////////

FrameSeq::FrameSeq(const TensorList &data, const PixelInfo &pix_info)
    : pix_info_(pix_info), pix_desc_(pix_info.format())
{
    HMP_REQUIRE(pix_desc_.defined(),
         "Unsupported PixelFormat {}", pix_info.format());

    for(auto &d : data){
        if(d.dim() == 3){
            data_.push_back(d.unsqueeze(-1)); //add channel dim
        }
        else{
            data_.push_back(d.alias());
        }
    }
}

FrameSeq::operator bool() const
{
    return data_.size() > 0;
}

FrameSeq FrameSeq::to(const Device &device, bool non_blocking) const
{
    TensorList out;
    for(auto &d : data_){
        out.push_back(d.to(device, non_blocking));
    }
    return FrameSeq(out, pix_info_);
}


FrameSeq FrameSeq::to(DeviceType device, bool non_blocking) const
{
    TensorList out;
    for(auto &d : data_){
        out.push_back(d.to(device, non_blocking));
    }
    return FrameSeq(out, pix_info_);
}


FrameSeq FrameSeq::crop(int left, int top, int w, int h) const
{
    auto width = this->width();
    auto height = this->height();
    left = wrap_size(left, width);
    top = wrap_size(top, height);

    auto right = left + w;
    auto bottom = top + h;
    HMP_REQUIRE(left < right && right <= width, 
        "FrameSeq::crop expect left({}) < right({}) and right <= {}", left, right, width);
    HMP_REQUIRE(top < bottom && bottom <= height, 
        "FrameSeq::crop expect top({}) < bottom({}) and bottom <= {}", top, bottom, height);

    //normalize
    double left_d = double(left)/width;
    double right_d = double(right)/width;
    double bottom_d = double(bottom)/height;
    double top_d = double(top)/height;

    //
    TensorList out;
    for(auto &d : data_){
        auto l = int(std::round(left_d*d.size(2)));
        auto r = int(std::round(right_d*d.size(2)));
        auto b = int(std::round(bottom_d*d.size(1)));
        auto t = int(std::round(top_d*d.size(1)));

        out.push_back(d.slice(1, t, b).slice(2, l, r));
    }

    return FrameSeq(out, pix_info_);
}

Frame FrameSeq::operator[](int64_t index) const
{
    HMP_REQUIRE(index < batch(), "FrameSeq: index out of range");
    TensorList planes;
    for(auto &d : data_){
        planes.push_back(d.select(0, index));
    }

    return Frame(planes, pix_info_);
}

FrameSeq FrameSeq::slice(int64_t start, optional<int64_t> end) const
{
    TensorList data;
    for(auto &d : data_){
        data.push_back(d.slice(0, start, end));
    }
    return FrameSeq(data, pix_info_);
}


//Tensor FrameSeq::to_rgb(ChannelFormat cformat) const
//{
//    return to_image(cformat).data();
//}

//ImageSeq FrameSeq::to_image(ChannelFormat cformat) const
//{
//    if(pix_info_.is_rgbx()){
//        HMP_REQUIRE(data_.size() == 1, "Internal error");
//        return ImageSeq(data_[0], kNHWC, pix_info_.color_model()).to(cformat);
//    }
//    else{
//        HMP_REQUIRE(pix_desc_.defined(),
//                    "FrameSeq::to_image: pixel format {} is not supported",
//                     pix_desc_.format());
//        auto data = img::yuv_to_rgb(data_, pix_info_, cformat);
//        return ImageSeq(data, cformat, pix_info_.color_model());
//    }
//}


FrameSeq FrameSeq::resize(int width, int height, ImageFilterMode mode) const
{
    if(!pix_info_.is_rgbx()){
        auto wscale = double(width) / this->width();
        auto hscale = double(height) / this->height(); 

        TensorList out;
        for(auto &d : data_){
            auto w = int64_t(std::round(d.size(2) * wscale));
            auto h = int64_t(std::round(d.size(1) * hscale));
            auto shape = d.shape();
            shape[d.dim() - 2] = w;
            shape[d.dim() - 3] = h;
            out.push_back(empty(shape, d.options()));
        }

        out = img::yuv_resize(out, data_, pix_info_, mode);
        return FrameSeq(out, pix_info_);
    }
    else{
        auto out = img::resize(data_[0], width, height, mode, kNHWC);
        return FrameSeq({out}, pix_info_);
    }
}


FrameSeq FrameSeq::rotate(ImageRotationMode mode) const
{
    if(!pix_info_.is_rgbx()){
        bool shapeChanged = mode == ImageRotationMode::Rotate90 || mode == ImageRotationMode::Rotate270;
        TensorList out;
        for(auto &d : data_){
            auto shape = d.shape();
            if(shapeChanged){
                std::swap(shape[d.dim()-2], shape[d.dim()-3]);
            }
            out.push_back(empty(shape, d.options()));
        }

        out = img::yuv_rotate(out, data_, pix_info_, mode);
        return FrameSeq(out, pix_info_);
    }
    else{
        auto out = img::rotate(data_[0], mode, kNHWC);
        return FrameSeq({out}, pix_info_);
    }
}


FrameSeq FrameSeq::mirror(ImageAxis axis) const
{
    if(!pix_info_.is_rgbx()){
        TensorList out;
        for(auto &d: data_){
            out.push_back(empty_like(d, d.options()));
        }
        out = img::yuv_mirror(out, data_, pix_info_, axis);
        return FrameSeq(out, pix_info_);
    }
    else{
        auto out = img::mirror(data_[0], axis, kNHWC);
        return FrameSeq({out}, pix_info_);
    }
}

FrameSeq FrameSeq::reformat(const PixelInfo &pix_info)
{
    if (pix_info_.format() == PF_RGB24) {
        auto yuv = img::rgb_to_yuv(data_[0], pix_info, kNHWC);
        return FrameSeq(yuv, pix_info);

    } else if (pix_info.format() == PF_RGB24) {
        auto rgb = img::yuv_to_rgb(data_, pix_info_, kNHWC);
        return FrameSeq({rgb}, pix_info);
    }

    HMP_REQUIRE(false, "{} to {} not support", stringfy(pix_info_.format()), stringfy(pix_info.format()));
}


FrameSeq concat(const std::vector<Frame> &frames)
{
    HMP_REQUIRE(frames.size(), "Frame::concat: expect at least 1 frame");
    for(size_t i = 1; i < frames.size(); ++i){
        HMP_REQUIRE(frames[i].format() == frames[0].format(),
            "Frame::concat: got unexpect pixel format {} at {}, expect {}",
            frames[i].format(), i, frames[0].format());
        HMP_REQUIRE(frames[i].width() == frames[0].width() 
            && frames[i].height() == frames[0].height()
            && frames[i].nplanes() == frames[0].nplanes(),
            "Frame::concat: expect all frame have same shape");
    }

    //do tensor concat
    TensorList planes;
    for(int64_t p = 0; p < frames[0].nplanes(); ++p){
        TensorList tensors;
        for (size_t i = 0; i < frames.size(); ++i){
            tensors.push_back(frames[i].plane(p).unsqueeze(0)); //add batch dim
        }

        //concat along axis=0
        planes.push_back(hmp::concat(tensors, 0));
    }


    return FrameSeq(planes, frames[0].pix_info());
}

FrameSeq concat(const std::vector<FrameSeq> &frames)
{
    HMP_REQUIRE(frames.size(), "FrameSeq::concat: require frames.size() > 0");
    //check
    for(size_t i = 1; i < frames.size(); ++i){
        HMP_REQUIRE(frames[i].format() == frames[0].format(),
            "FrameSeq::concat expect all frame have same format {}, got {} at {}", frames[0].format(), frames[i].format(), i);
        HMP_REQUIRE(frames[i].width() == frames[0].width() && frames[i].height() == frames[0].height(),
            "FrameSeq::concat expect all frame have same size ({}, {}), got ({}, {}) at {}",
            frames[0].width(), frames[0].height(), frames[i].width(), frames[i].height(), i);
    }

    //do tensor concat
    TensorList planes;
    for(int64_t p = 0; p < frames[0].nplanes(); ++p){
        TensorList tensors;
        for (size_t i = 0; i < frames.size(); ++i){
            tensors.push_back(frames[i].plane(p));
        }

        //concat along axis=0
        planes.push_back(hmp::concat(tensors, 0));
    }

    return FrameSeq(planes, frames[0].pix_info());
}

std::string stringfy(const FrameSeq &frames)
{
    return fmt::format("FrameSeq({}, {}, {}, ({}, {}, {}, {}))",
        frames.device(), frames.dtype(), frames.format(),
        frames.batch(), frames.nplanes(), frames.height(), frames.width());
}

} //
