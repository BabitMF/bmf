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
#include <hmp/cv2/cv2_helper.h>

#include <opencv2/imgproc.hpp>
#ifdef HMP_ENABLE_CUDA
#include <opencv2/cudafilters.hpp>
#endif


namespace hmp{
namespace kernel{
namespace ocv{

using hmp::ocv::to_cv_mat;
using hmp::ocv::from_cv_mat;
using hmp::ocv::to_cv_type;
using hmp::ocv::from_cv_type;

inline int to_cv_filter_mode(ImageFilterMode mode)
{
    switch(mode){
        case kNearest: return cv::INTER_NEAREST;
        case kBilinear: return cv::INTER_LINEAR;
        case kBicubic: return cv::INTER_CUBIC;
        default:
            HMP_REQUIRE(false, "{} is not supported by OpenCV", mode);
    }
}


template<typename Func, typename ...Args>
void foreach_image(const Func &f, ChannelFormat cformat, Tensor &dst, Args&&...args)
{
    auto batch = dst.size(0);
    for(int64_t i = 0; i < batch; ++i){
        if(cformat == ChannelFormat::NCHW){
            for(int64_t c = 0; c < dst.size(1); ++c){
                auto dmat = to_cv_mat(dst.select(0, i).select(0, c), false);
                f(dmat, to_cv_mat(args.select(0, i).select(0, c), false)...);
            }
        }
        else{
            auto dmat = to_cv_mat(dst.select(0, i), true);
            f(dmat, to_cv_mat(args.select(0, i), true)...);
        }
    }
}


static inline void morph(
    cv::MorphTypes algo, const cv::Mat &src, cv::Mat &dst, cv::Mat &kernel)
{
    switch(algo){
        case cv::MORPH_ERODE: 
            cv::erode(src, dst, kernel);
            break;
        case cv::MORPH_DILATE:
            cv::dilate(src, dst, kernel);
            break;
        default:
            HMP_REQUIRE(false, "MorphType {} not implemeted", algo);
    }
}


}}} //namespace hmp::kernel::ocv