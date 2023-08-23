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
#include <bmf/sdk/convert_backend.h>
#include <bmf/sdk/sdk_interface.h>
#include <bmf/sdk/log.h>
#include <bmf/sdk/json_param.h>
#include <unordered_map>
#include <bmf/sdk/video_filter.h>

namespace bmf_sdk {

static std::unordered_map<MediaType, Convertor *> iConvertors;

BMF_API void set_convertor(const MediaType &media_type, Convertor *convertor) {
    iConvertors[media_type] = convertor;
}

BMF_API Convertor *get_convertor(const MediaType &media_type) {
    if (iConvertors.find(media_type) == iConvertors.end()) {
        BMFLOG(BMF_WARNING) << "the media type is not supported by bmf backend";
        return NULL;
    }
    return iConvertors[media_type];
}

BMF_API VideoFrame bmf_convert(VideoFrame &vf, const MediaDesc &dp) {
    auto convt = get_convertor(MediaType::kBMFVideoFrame);
    if (dp.media_type.has_value()) {
        convt = get_convertor(dp.media_type());
    }

    auto format_vf = convt->format_cvt(vf, dp);
    auto device_vf = convt->device_cvt(format_vf, dp);
    convt->media_cvt(device_vf, dp);
    return device_vf;
}

BMF_API VideoFrame bmf_convert_to_videoframe(VideoFrame &vf,
                                             const MediaDesc &dp) {
    VideoFrame frame;
    auto convt = get_convertor(MediaType::kBMFVideoFrame);
    if (dp.media_type.has_value()) {
        convt = get_convertor(dp.media_type());
    }

    int ret = convt->media_cvt_to_videoframe(vf, dp);
    if (ret != 0) {
        return frame;
    }
    return vf;
}

// media type is kBMFVideoFrame
bool mt_is_vf(const MediaDesc &dp) {
    return !(dp.media_type.has_value() &&
             dp.media_type() != MediaType::kBMFVideoFrame);
}

BMF_API VideoFrame bmf_convert(VideoFrame &src_vf, const MediaDesc &src_dp,
                               const MediaDesc &dst_dp) {
    VideoFrame frame;
    // VideoFrame transfer to other mediatype
    if (mt_is_vf(src_dp)) {
        frame = bmf_convert(src_vf, dst_dp);

    } else if (mt_is_vf(dst_dp)) {
        frame = bmf_convert_to_videoframe(src_vf, src_dp);
        frame = bmf_convert(frame, dst_dp);

    } else {
        BMFLOG(BMF_ERROR) << "can not tranfer from src type: "
                          << static_cast<int>(src_dp.media_type())
                          << " to dst type: "
                          << static_cast<int>(dst_dp.media_type());
    }

    return frame;
}

Convertor::Convertor() {}

VideoFrame Convertor::format_cvt(VideoFrame &src, const MediaDesc &dp) {
    VideoFrame dst = src;
    // do scale
    if (dp.width.has_value() || dp.height.has_value()) {
        int w = 0;
        int h = 0;
        if (!dp.width.has_value()) {
            h = dp.height();
            w = src.width() * h / src.height();

        } else if (!dp.height.has_value()) {
            w = dp.width();
            h = src.height() * w / src.width();

        } else {
            w = dp.width();
            h = dp.height();
        }

        dst = bmf_scale_func_with_param(dst, w, h);
    }

    // csc

    if (dp.pixel_format.has_value()) {
        if (dp.color_space.has_value()) {
            dst = bmf_csc_func_with_param(
                dst, hmp::PixelInfo(dp.pixel_format(), dp.color_space()));

        } else {
            dst =
                bmf_csc_func_with_param(dst, hmp::PixelInfo(dp.pixel_format()));
        }
    }

    return dst;
}

VideoFrame Convertor::device_cvt(VideoFrame &src, const MediaDesc &dp) {
    if (dp.device.has_value()) {
        return src.to(dp.device());
    }
    return src;
}

int Convertor::media_cvt(VideoFrame &src, const MediaDesc &dp) { return 0; }

int Convertor::media_cvt_to_videoframe(VideoFrame &src, const MediaDesc &dp) {
    return 0;
}

static Convertor *iDefaultBMFConvertor = new Convertor();

BMF_REGISTER_CONVERTOR(MediaType::kBMFVideoFrame, iDefaultBMFConvertor);

} // namespace bmf_sdk
