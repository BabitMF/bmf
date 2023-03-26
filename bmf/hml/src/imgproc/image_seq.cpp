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


Tensor FrameSeq::to_rgb(ChannelFormat cformat) const
{
    return to_image(cformat).data();
}

ImageSeq FrameSeq::to_image(ChannelFormat cformat) const
{
    if(pix_info_.is_rgbx()){
        HMP_REQUIRE(data_.size() == 1, "Internal error");
        return ImageSeq(data_[0], kNHWC, pix_info_.color_model()).to(cformat);
    }
    else{
        HMP_REQUIRE(pix_desc_.defined(),
                    "FrameSeq::to_image: pixel format {} is not supported",
                     pix_desc_.format());
        auto data = img::yuv_to_rgb(data_, pix_info_, cformat);
        return ImageSeq(data, cformat, pix_info_.color_model());
    }
}


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


FrameSeq FrameSeq::from_rgb(const Tensor &rgb, const PixelInfo &pix_info, ChannelFormat cformat)
{
    ImageSeq images(rgb, cformat, pix_info.color_model());
    return from_image(images, pix_info);
}

FrameSeq FrameSeq::from_image(const ImageSeq &images, const PixelInfo &pix_info)
{
    if(pix_info.is_rgbx()){
        auto pix_desc = PixelFormatDesc(pix_info.format());
        HMP_REQUIRE(pix_desc.defined() && pix_desc.channels(0) == images.nchannels(), 
            "FrameSeq::from_image: expect image has {} channels, got {}",
            pix_desc.channels(), images.nchannels());
        HMP_REQUIRE(images.format() == kNHWC, 
            "FrameSeq::from_image: expect image has NHWC layout");
        HMP_REQUIRE(images.dtype() == pix_desc.dtype(),
            "FrameSeq:from_image: expect image has dtype {}, got {}",
            pix_desc.dtype(), images.dtype());
        return FrameSeq({images.data()}, pix_info);
    }
    else{
        auto yuv = img::rgb_to_yuv(images.data(), pix_info, images.format());
        return FrameSeq(yuv, pix_info);
    }
}

/////////////////////////////////// ImageSeq ///////////////////////////


ImageSeq::ImageSeq(const Tensor &data, ChannelFormat format, const ColorModel &cm)
    : cm_(cm)
{
    HMP_REQUIRE(data.dim() >= 3,
        "ImageSeq: invalid ImageSeq data, got {}", data.dim());

    if(data.dim() == 3){
        if(format_ == kNCHW){
            data_ = data.unsqueeze(1);
        }
        else{
            data_ = data.unsqueeze(-1);
        }
    }
    else{
        data_ = data.alias();
    }

    format_ = format;
}


ImageSeq::operator bool() const
{
    return data_.defined();
}

ImageSeq ImageSeq::to(const Device &device, bool non_blocking) const
{
    auto data = data_.to(device, non_blocking);
    return ImageSeq(data, format_, cm_);
}

ImageSeq ImageSeq::to(DeviceType device, bool non_blocking) const
{
    auto data = data_.to(device, non_blocking);
    return ImageSeq(data, format_, cm_);
}

ImageSeq ImageSeq::to(ScalarType dtype) const
{
    auto data = data_.to(dtype);
    return ImageSeq(data, format_, cm_);
}

ImageSeq ImageSeq::to(ChannelFormat format, bool contiguous) const
{
    Tensor data = data_;
    if(format == ChannelFormat::NCHW && format_ == ChannelFormat::NHWC){
        data = data_.permute({0, 3, 1, 2});
    }
    else if(format == ChannelFormat::NHWC && format_ == ChannelFormat::NCHW){
        data = data_.permute({0, 2, 3, 1});
    }

    if(contiguous){
        data = data.contiguous();
    }
    return ImageSeq(data, format, cm_);
}

ImageSeq& ImageSeq::copy_(const ImageSeq &from)
{
    HMP_REQUIRE(data().shape() == from.data().shape(), 
        "ImageSeq: data shaped are not matched, expect {}, got {}",
        data().shape(), from.data().shape());

    copy(data_, from.data());
    return *this;
}


ImageSeq ImageSeq::crop(int left, int top, int w, int h) const
{
    auto width = this->width();
    auto height = this->height();
    left = wrap_size(left, width);
    top = wrap_size(top, height);

    auto right = left + w;
    auto bottom = top + h;
    HMP_REQUIRE(left < right && right <= width, 
        "ImageSeq::crop expect left({}) < right({}) and right <= {}", left, right, width);
    HMP_REQUIRE(top < bottom && bottom <= height, 
        "ImageSeq::crop expect top({}) < bottom({}) and bottom <= {}", top, bottom, height);

    auto data = data_.slice(wdim(), left, right).slice(hdim(), top, bottom);
    return ImageSeq(data, format_, cm_);
}


Image ImageSeq::operator[](int64_t index) const
{
    HMP_REQUIRE(index < batch(), "ImageSeq: index is out of range");
    return Image(data_.select(0, index), format(), cm_);
}


ImageSeq ImageSeq::slice(int64_t start, optional<int64_t> end_) const
{
    auto end = end_.value_or(data_.size(0)); 
    auto data = data_.slice(0, start, end);
    return ImageSeq(data, format_, cm_);
}


ImageSeq ImageSeq::resize(int width, int height, ImageFilterMode mode) const
{
    auto shape = data_.shape();
    shape[wdim()] = width;
    shape[hdim()] = height;
    auto data = empty(shape, data_.options());

    img::resize(data, data_, mode, format_);

    return ImageSeq(data, format_, cm_);
}

ImageSeq ImageSeq::select(int channel) const
{
    auto data = data_.slice(cdim(), channel, channel+1);
    return ImageSeq(data, format_, cm_); 
}

ImageSeq ImageSeq::rotate(ImageRotationMode mode) const
{
    auto width = this->width();
    auto height = this->height();
    bool shapeChanged = mode == ImageRotationMode::Rotate90 || mode == ImageRotationMode::Rotate270;
    if(shapeChanged){
        std::swap(width, height);
    }

    auto shape = data_.shape();
    shape[wdim()] = width;
    shape[hdim()] = height;
    auto data = empty(shape, data_.options());

    img::rotate(data, data_, mode, format_);

    return ImageSeq(data, format_, cm_);
}


ImageSeq ImageSeq::mirror(ImageAxis axis) const
{
    auto out = empty_like(data_, data_.options());
    img::mirror(out, data_, axis, format_);
    return ImageSeq(out, format_, cm_);
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


ImageSeq concat(const std::vector<Image> &images)
{
    //check
    HMP_REQUIRE(images.size(), "Image::concat expect at least 1 image");
    for(size_t i = 1; i < images.size(); ++i){
        HMP_REQUIRE(images[i].format() == images[0].format(),
            "Image::concat expect all images have same format {}, got {} at {}",
            images[0].format(), images[i].format(), i);
        HMP_REQUIRE(images[i].width() == images[0].width() 
                   && images[i].height() == images[0].height() 
                   && images[i].nchannels() == images[0].nchannels(),
            "Image::concat expect all frame have same size {}, got {} at {}",
            images[0], images[i], i);
    }

    //
    TensorList tensors;
    for(auto &im : images){
        tensors.push_back(im.data().unsqueeze(0)); //add batch dim
    }
    auto data = hmp::concat(tensors, 0);


    return ImageSeq(data, images[0].format(), images[0].color_model());
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



ImageSeq concat(const std::vector<ImageSeq> &images)
{
    //check
    HMP_REQUIRE(images.size(), "ImageSeq::concat expect at least 1 image");
    for(size_t i = 1; i < images.size(); ++i){
        HMP_REQUIRE(images[i].format() == images[0].format(),
            "ImageSeq::concat expect all images have same format {}, got {} at {}",
            images[0].format(), images[i].format(), i);
        HMP_REQUIRE(images[i].width() == images[0].width() 
                   && images[i].height() == images[0].height() 
                   && images[i].nchannels() == images[0].nchannels(),
            "ImageSeq::concat expect all frame have same size {}, got {} at {}",
            images[0], images[i], i);
    }

    //
    TensorList tensors;
    for(auto &im : images){
        tensors.push_back(im.data());
    }
    auto data = hmp::concat(tensors);

    return ImageSeq(data, images[0].format(), images[0].color_model());
}



std::string stringfy(const FrameSeq &frames)
{
    return fmt::format("FrameSeq({}, {}, {}, ({}, {}, {}, {}))",
        frames.device(), frames.dtype(), frames.format(),
        frames.batch(), frames.nplanes(), frames.height(), frames.width());
}

std::string stringfy(const ImageSeq &images)
{
    return fmt::format("ImageSeq({}, {}, {}, {})",
        images.device(), images.dtype(), images.format(),
        images.data().shape());
}


} //