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

#include <bmf/sdk/sdk_interface.h>
#include <bmf/sdk/log.h>
#include <bmf/sdk/convert_backend.h>
#include <hmp/cv2/cv2_helper.h>

namespace bmf_sdk {

template <> struct OpaqueDataInfo<cv::Mat> {
    const static int key = OpaqueDataKey::kCVMat;
    static OpaqueData construct(const cv::Mat *mat) {
        return std::make_shared<cv::Mat>(*mat);
    }
};

class OCVConvertor : public Convertor {
  public:
    OCVConvertor() {}
    int media_cvt(VideoFrame &src, const MediaDesc &dp) override {
        try {
            if (!src.frame().pix_info().is_rgbx()) {
                BMFLOG(BMF_ERROR) << "cvmat only support rgbx frame";
                return -1;
            }
            auto tensor = src.frame().data()[0];
            cv::Mat mat = hmp::ocv::to_cv_mat(tensor, true);
            cv::Mat *pmat = new cv::Mat(mat);
            src.private_attach<cv::Mat>(pmat);

        } catch (std::exception &e) {
            BMFLOG(BMF_ERROR) << "convert to cv::mat err: " << e.what();
            return -1;
        }
        return 0;
    }

    int media_cvt_to_videoframe(VideoFrame &src, const MediaDesc &dp) override {
        try {
            if (!dp.pixel_format.has_value()) {
                BMFLOG(BMF_ERROR) << "VideoFrame format represented by the "
                                     "cv::Mat must be specified.";
                return -1;
            }

            const cv::Mat *pmat = src.private_get<cv::Mat>();
            if (!pmat) {
                BMFLOG(BMF_ERROR) << "private data is null, please use "
                                     "private_attach before call this api";
                return -1;
            }

            Tensor tensor = hmp::ocv::from_cv_mat(*pmat);
            VideoFrame vf(Frame(tensor, hmp::PixelInfo(dp.pixel_format())));
            vf.private_attach<cv::Mat>(pmat);
            src = vf;

        } catch (std::exception &e) {
            BMFLOG(BMF_ERROR) << "convert to cv::mat err: " << e.what();
            return -1;
        }
        return 0;
    }
};

static Convertor *ocv_convert = new OCVConvertor();
BMF_REGISTER_CONVERTOR(MediaType::kCVMat, ocv_convert);
} // namespace bmf_sdk
