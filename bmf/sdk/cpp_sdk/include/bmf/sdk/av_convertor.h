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

#include <bmf/sdk/convert_backend.h>
#include <bmf/sdk/ffmpeg_helper.h>

namespace bmf_sdk {

class AVConvertor : public Convertor {
  public:
    AVConvertor() {}
    int media_cvt(VideoFrame &src, const MediaDesc &dp) override {
        try {
            AVFrame *frame = ffmpeg::from_video_frame(src, false);
            src.private_attach<AVFrame>(frame);
        } catch (std::exception &e) {
            BMFLOG(BMF_ERROR) << "convert to AVFrame err: " << e.what();
            return -1;
        }
        return 0;
    }

    int media_cvt_to_videoframe(VideoFrame &src, const MediaDesc &dp) override {
        try {
            const AVFrame *avf = src.private_get<AVFrame>();
            if (!avf) {
                BMFLOG(BMF_ERROR) << "private data is null, please use "
                                     "private_attach before call this api";
                return -1;
            }
            VideoFrame res = ffmpeg::to_video_frame(avf, true);
            src = res;
        } catch (std::exception &e) {
            BMFLOG(BMF_ERROR)
                << "AVFrame convert to VideoFrame err: " << e.what();
            return -1;
        }
        return 0;
    }
};

static Convertor *av_convert = new AVConvertor();
BMF_REGISTER_CONVERTOR(MediaType::kAVFrame, av_convert);
} // namespace bmf_sdk
