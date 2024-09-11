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

#ifndef _BMF_LITE_VIDEOFRAME_H_
#define _BMF_LITE_VIDEOFRAME_H_

#include "common/bmf_common.h"
#include "common/bmf_param.h"
#include "media/video_buffer/video_buffer.h"
#include <memory>
#include <string>
#include <vector>

namespace bmf_lite {

enum {
    BMF_LITE_PIXEL_FORMAT_RGBA = 0,
    BMF_LITE_PIXEL_FORMAT_YUV420P = 100,
    BMF_LITE_PIXEL_FORMAT_TEXTURE2D = 200,
    BMF_LITE_PIXEL_FORMAT_OES_TEXTURE = 300,
    BMF_LITE_PIXEL_FORMAT_CVPIXEL_BUFFER = 400
};

class VideoFrameImpl;

class BMF_LITE_EXPORT VideoFrame {

  public:
    VideoFrame();
    VideoFrame(std::shared_ptr<VideoBuffer> video_buffer);

    ~VideoFrame();
    std::shared_ptr<VideoBuffer> buffer();

  protected:
    std::shared_ptr<VideoFrameImpl> impl_ = nullptr;
};

} // namespace bmf_lite

#endif // _BMF_LITE_VIDEOFRAME_H_