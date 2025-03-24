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

namespace hmp {

Frame::Frame(const TensorList &data, int width, int height,
             const PixelInfo &pix_info, const Tensor &storage_tensor)
    : pix_info_(pix_info), width_(width), height_(height),
      storage_tensor_(storage_tensor) {
    pix_desc_ = PixelFormatDesc(pix_info_.format());

    // convert to HWC layout if pixel format is supported
    if (pix_desc_.defined()) {
        data_ = img::frame_format(data, pix_desc_, width, height);
    } else {
        data_ = data;
    }
}

Frame::Frame(const TensorList &data, const PixelInfo &pix_info,
             const Tensor &storage_tensor)
    : Frame(data, data[0].size(1), data[0].size(0), pix_info, storage_tensor) {}

Frame::Frame(const Tensor &data, const PixelInfo &pix_info,
             const Tensor &storage_tensor)
    : Frame({data}, data.size(1), data.size(0), pix_info, storage_tensor) {}

Frame::Frame(int width, int height, const PixelInfo &pix_info,
             const Device &device)
    : pix_info_(pix_info), width_(width), height_(height) {
    pix_desc_ = PixelFormatDesc(pix_info_.format());

    HMP_REQUIRE(pix_desc_.defined(), "PixelFormat {} is not supported by hmp",
                pix_info_.format());

    auto options = TensorOptions(device).dtype(pix_desc_.dtype());
    // calculate total size of frame

    int64_t nitems = 0;

    SizeArray offsetArray;

    for (int i = 0; i < pix_desc_.nplanes(); ++i) {
        SizeArray shape{pix_desc_.infer_height(height, i),
                        pix_desc_.infer_width(width, i), pix_desc_.channels(i)};

        offsetArray.push_back(nitems);
        nitems += TensorInfo::calcNumel(shape);
    }

    storage_tensor_ = empty({1, nitems}, options);

    for (int i = 0; i < pix_desc_.nplanes(); ++i) {
        SizeArray shape{pix_desc_.infer_height(height, i),
                        pix_desc_.infer_width(width, i), pix_desc_.channels(i)};
        auto strides = calcContiguousStrides(shape);
        Tensor t = storage_tensor_.as_strided(shape, strides, offsetArray[i]);
        data_.push_back(t);
    }
}

Frame Frame::to(const Device &device, bool non_blocking) const {
    if (storage_tensor_.defined()) {
        Tensor new_storage_tensor = storage_tensor_.to(device, non_blocking);
        TensorList out =
            from_storage_tensor(new_storage_tensor, data_); // change buffer
        return Frame(out, width_, height_, pix_info_, new_storage_tensor);
    }

    //
    TensorList out;
    for (auto &d : data_) {
        out.push_back(d.to(device, non_blocking));
    }
    return Frame(out, width_, height_, pix_info_);
}

Frame Frame::to(DeviceType device, bool non_blocking) const {
    return to(Device(device), non_blocking);
}

Frame &Frame::copy_(const Frame &from) {
    HMP_REQUIRE(format() == from.format(),
                "Can't copy from different PixelFormat {}, expect {}",
                from.format(), format());
    for (size_t i = 0; i < data_.size(); ++i) {
        data_[i].copy_(from.data_[i]);
    }
    return *this;
}

Frame Frame::clone() const {
    if (storage_tensor_.defined()) {
        Tensor new_storage_tensor = storage_tensor_.clone();
        TensorList out =
            from_storage_tensor(new_storage_tensor, data_); // change buffer
        return Frame(out, width_, height_, pix_info_, new_storage_tensor);
    }

    TensorList out;
    for (auto &d : data_) {
        out.push_back(d.clone());
    }
    return Frame(out, width_, height_, pix_info_);
}

Frame Frame::crop(int left, int top, int w, int h) const {
    HMP_REQUIRE(pix_desc_.defined(),
                "Frame::crop: pixel format {} is not supported",
                pix_desc_.format());

    auto width = this->width();
    auto height = this->height();
    left = wrap_size(left, width);
    top = wrap_size(top, height);

    auto right = left + w;
    auto bottom = top + h;
    HMP_REQUIRE(left < right && right <= width,
                "Frame::crop expect left({}) < right({}) and right <= {}", left,
                right, width);
    HMP_REQUIRE(top < bottom && bottom <= height,
                "Frame::crop expect top({}) < bottom({}) and bottom <= {}", top,
                bottom, height);
    //
    TensorList out;
    int plane_w, plane_h, plane_x, plane_y;
    for(int i = 0; i < data_.size(); ++i) {
        plane_w = pix_desc_.infer_width(w, i);
        plane_h = pix_desc_.infer_height(h, i);
        plane_x = pix_desc_.infer_width(left, i);
        plane_y = pix_desc_.infer_height(top, i);

        auto l = plane_x;
        auto r = plane_x + plane_w;
        auto t = plane_y;
        auto b = plane_y + plane_h;

        auto &d = data_[i];
        out.push_back(d.slice(0, t, b).slice(1, l, r));
    }

    return Frame(out, w, h, pix_info_, storage_tensor_);
}

PixelFormat format420_list[] = {PF_YUV420P, PF_NV12, PF_NV21, PF_P010LE,
                                PF_YUV420P10LE};
static bool is_420(const PixelFormat &pix_fmt) {
    for (auto i : format420_list) {
        if (pix_fmt == i)
            return true;
    }
    return false;
}

Frame Frame::reformat(const PixelInfo &pix_info) {
    if (pix_info_.format() == PF_RGB24 || pix_info_.format() == PF_RGB48) {
        Frame frame(width_, height_, pix_info, device());
        auto yuv = img::rgb_to_yuv(frame.data(), data_[0], pix_info, kNHWC);
        return frame;
    } else if (pix_info_.format() == PF_BGR24 || pix_info_.format() == PF_BGR48) {
        auto yuv = img::bgr_to_yuv(data_[0], pix_info, kNHWC);
        return Frame(yuv, width_, height_, pix_info);
    } else if (pix_info.format() == PF_BGR24 || pix_info.format() == PF_BGR48) {
        auto rgb = img::yuv_to_bgr(data_, pix_info_, kNHWC);
        return Frame({rgb}, width_, height_, pix_info);
    } else if (pix_info.format() == PF_RGB24 || pix_info.format() == PF_RGB48) {
        auto rgb = img::yuv_to_rgb(data_, pix_info_, kNHWC);
        return Frame({rgb}, width_, height_, pix_info);

    } else if (is_420(pix_info.format()) && is_420(pix_info_.format())) {
        Frame frame(width_, height_, pix_info, device());
        auto yuv = img::yuv_to_yuv(frame.data(), data_, pix_info, pix_info_);
        return frame;
    }

    HMP_REQUIRE(false, "{} to {} not support", stringfy(pix_info_.format()),
                stringfy(pix_info.format()));
}

TensorList from_storage_tensor(const Tensor &storage_tensor,
                               const TensorList &mirror) {
    TensorList out; // change buffer

    if (!storage_tensor.defined()) {
        return out;
    }

    Buffer b = storage_tensor.tensorInfo()->buffer();

    for (auto &d : mirror) {
        auto old_ti = d.tensorInfo();
        auto ti = makeRefPtr<TensorInfo>(b, old_ti->shape(), old_ti->strides(),
                                         old_ti->bufferOffset());
        Tensor t(std::move(ti));
        out.push_back(t);
    }

    return out;
}

Frame Frame::as_contiguous_storage() {
    if (storage_tensor_.defined()) {
        return *this;
    }
    Frame frame(width_, height_, pix_info_, device());
    frame.copy_(*this);
    return frame;
}

std::string stringfy(const Frame &frame) {
    return fmt::format("Frame({}, {}, {}, ({}, {}, {}))", frame.device(),
                       frame.dtype(), frame.format(), frame.nplanes(),
                       frame.height(), frame.width());
}

} // namespace hmp
