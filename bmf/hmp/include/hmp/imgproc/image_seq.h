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
#include <hmp/imgproc/image.h>
#include <hmp/imgproc/formats.h>

namespace hmp {

class HMP_API FrameSeq {
  public:
    FrameSeq() = default;
    FrameSeq(const FrameSeq &) = default;
    FrameSeq(const TensorList &data, const PixelInfo &pix_info);

    operator bool() const;

    const PixelFormatDesc &pix_desc() const { return pix_desc_; }
    const PixelInfo &pix_info() const { return pix_info_; }
    PixelFormat format() const { return pix_info_.format(); }
    int batch() const { return data_[0].size(0); }
    int width() const { return data_[0].size(2); }
    int height() const { return data_[0].size(1); }
    ScalarType dtype() const { return data_[0].dtype(); }
    Device device() const { return data_[0].device(); }

    int64_t nplanes() const { return data_.size(); }
    const Tensor &plane(int64_t p) const { return data_[p]; }
    const TensorList &data() const { return data_; }

    FrameSeq to(const Device &device, bool non_blocking = false) const;
    FrameSeq to(DeviceType device, bool non_blocking = false) const;

    Frame operator[](int64_t index) const;
    FrameSeq slice(int64_t start, optional<int64_t> end = nullopt) const;
    FrameSeq crop(int left, int top, int width, int height) const;
    FrameSeq resize(int width, int height,
                    ImageFilterMode mode = ImageFilterMode::Bicubic) const;
    FrameSeq rotate(ImageRotationMode mode) const;
    FrameSeq mirror(ImageAxis axis = ImageAxis::Horizontal) const;

    FrameSeq reformat(const PixelInfo &pix_info);

  private:
    PixelFormatDesc pix_desc_;
    PixelInfo pix_info_;
    TensorList data_;
};

HMP_API FrameSeq concat(const std::vector<Frame> &frames);
HMP_API FrameSeq concat(const std::vector<FrameSeq> &frames);

HMP_API std::string stringfy(const FrameSeq &frames);

} // namespace hmp
