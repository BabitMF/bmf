/*
 * Copyright 2024 Babit Authors
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

#include "algorithm/bmf_video_frame.h"
#include "media/video_buffer/video_buffer.h"

namespace bmf_lite {

class VideoFrameImpl {
  public:
    VideoFrameImpl(std::shared_ptr<VideoBuffer> video_buffer) {
        buffer_ = video_buffer;
    }

    VideoFrameImpl() {}

    std::shared_ptr<VideoBuffer> buffer_;
};

VideoFrame::VideoFrame() { impl_ = std::make_shared<VideoFrameImpl>(); }

VideoFrame::VideoFrame(std::shared_ptr<VideoBuffer> video_buffer) {
    impl_ = std::make_shared<VideoFrameImpl>(video_buffer);
}

std::shared_ptr<VideoBuffer> VideoFrame::buffer() { return impl_->buffer_; }

VideoFrame::~VideoFrame() {}

} // namespace bmf_lite