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
#include <bmf/sdk/video_filter.h>
#include <bmf/sdk/log.h>

#include <hmp/imgproc.h>

BEGIN_BMF_SDK_NS

#define GET_OR_RETURN(key, value, key_type, return_value)                      \
    do {                                                                       \
        if (!(param.has_key(key))) {                                           \
            BMFLOG(BMF_ERROR) << "get " << key << " failed";                   \
            return return_value;                                               \
        }                                                                      \
        param.get_##key_type(key, value);                                      \
    } while (0)

VideoFrame bmf_scale_func_with_param(VideoFrame &src_vf, int w, int h,
                                     int mode) {
    VideoFrame frame;
    PixelInfo pix_info = src_vf.frame().pix_info();

    if (src_vf.width() == w && src_vf.height() == h) {
        return src_vf;
    }

    frame = VideoFrame::make(w, h, pix_info, src_vf.device());

    auto scale_mode = static_cast<hmp::ImageFilterMode>(mode);

    // check support
    if (src_vf.frame().pix_info().is_rgbx()) {
        hmp::Tensor dst_tensor = frame.frame().data()[0];
        hmp::img::resize(dst_tensor, src_vf.frame().data()[0], scale_mode,
                         kNHWC);

    } else {
        hmp::TensorList dst_tensor = frame.frame().data();
        hmp::img::yuv_resize(dst_tensor, src_vf.frame().data(), pix_info,
                             scale_mode);
    }

    // copy
    frame.copy_props(src_vf);
    return frame;
}

/** @addtogroup bmf_video_filter
 * @{
 * @arg width: dst frame width
 * @arg height: dst frame height
 * @arg mode: 0 is Nearest, 1 is Bilinear, 2 is Bicubic
 * @} */
VideoFrame bmf_scale_func(VideoFrame &src_vf, JsonParam param) {
    VideoFrame frame;
    int width;
    int height;
    int mode = 1;
    int result = 0;

    GET_OR_RETURN("width", width, int, frame);
    GET_OR_RETURN("height", height, int, frame);

    param.get_int("mode", mode);
    return bmf_scale_func_with_param(src_vf, width, height, mode);
}

VideoFrame bmf_csc_func_with_param(VideoFrame &src_vf,
                                   const hmp::PixelInfo &pixel_info) {
    // reformat
    return src_vf.reformat(pixel_info);
}

/** @addtogroup bmf_video_filter
 * @{
 * @arg pixfmt: dst frame pixfmt
 * @} */
VideoFrame bmf_csc_func(VideoFrame &src_vf, JsonParam param) {
    // use reformat
    VideoFrame frame;
    std::string pixfmt;
    GET_OR_RETURN("pixfmt", pixfmt, string, frame);

    hmp::PixelFormat pix_fmt = hmp::get_pixel_format(pixfmt);
    return bmf_csc_func_with_param(src_vf, pix_fmt);
}

REGISTER_VFFILTER(bmf_scale, bmf_scale_func)
REGISTER_VFFILTER(bmf_csc, bmf_csc_func)

END_BMF_SDK_NS
