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
#include <hmp/core/logging.h>
#include <bmf/sdk/video_frame.h>

namespace bmf_sdk {

struct VideoFrame::Private {
    Private(const Frame &frame_) : frame(frame_) {}

    Private(const Private &other) = default;

    Frame frame;
};

VideoFrame::VideoFrame() {
    // not defined
}

VideoFrame::VideoFrame(const Frame &frame) {
    self = std::make_shared<Private>(frame);
}

VideoFrame::VideoFrame(int width, int height, const PixelInfo &pix_info,
                       const Device &device)
    : VideoFrame(Frame(width, height, pix_info, device)) {}

VideoFrame::VideoFrame(const std::shared_ptr<Private> &other) : self(other) {}

VideoFrame::operator bool() const { return self.get() != nullptr; }

int VideoFrame::width() const { return self->frame.width(); }

int VideoFrame::height() const { return self->frame.height(); }

ScalarType VideoFrame::dtype() const { return self->frame.dtype(); }

const VideoFrame::Frame &VideoFrame::frame() const { return self->frame; }

VideoFrame VideoFrame::crop(int x, int y, int w, int h) const {
    VideoFrame vf;
    auto frame = self->frame.crop(x, y, w, h);
    vf = VideoFrame(frame);
    vf.copy_props(*this);
    return vf;
}

const Device &VideoFrame::device() const { return self->frame.device(); }

VideoFrame VideoFrame::cpu(bool non_blocking) const {
    VideoFrame vf;
    auto frame = self->frame.to(kCPU, non_blocking);
    vf = VideoFrame(frame);
    vf.copy_props(*this);

    return vf;
}

VideoFrame VideoFrame::VideoFrame::cuda() const {
    VideoFrame vf;
    auto frame = self->frame.to(kCUDA);
    vf = VideoFrame(frame);
    vf.copy_props(*this);

    return vf;
}

VideoFrame &VideoFrame::copy_(const VideoFrame &from) {
    self->frame.copy_(from.frame());
    return *this;
}

VideoFrame VideoFrame::to(const Device &device, bool non_blocking) const {
    VideoFrame vf;
    auto frame = self->frame.to(device, non_blocking);
    vf = VideoFrame(frame);
    vf.copy_props(*this);

    return vf;
}

VideoFrame &VideoFrame::copy_props(const VideoFrame &from, bool copy_private) {
    if (copy_private) {
        OpaqueDataSet::copy_props(from);
    }

    SequenceData::copy_props(from);
    Future::copy_props(from);
    return *this;
}

VideoFrame VideoFrame::reformat(const PixelInfo &pix_info) {
    auto frame = self->frame.reformat(pix_info);
    return VideoFrame(frame);
}

VideoFrame VideoFrame::as_contiguous_storage() {
    VideoFrame vf;
    auto frame = self->frame.as_contiguous_storage();
    vf = VideoFrame(frame);
    vf.SequenceData::copy_props(*this);
    return vf;
}

} // namespace bmf_sdk
