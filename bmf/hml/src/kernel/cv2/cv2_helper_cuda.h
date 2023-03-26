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

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <kernel/cv2/cv2_helper.h>
#include <kernel/cuda/kernel_utils.h>

namespace hmp{
namespace kernel{
namespace ocv{


static cv::cuda::GpuMat to_cv_gpu_mat(const Tensor &tensor, bool channel_last=false)
{
    HMP_REQUIRE(tensor.device_type() == kCUDA, "cv::GpuMat only support cuda tensor");
    HMP_REQUIRE(tensor.dim() - channel_last == 2,
         "cv::GpuMat require dim == 2, got {}", tensor.dim() - channel_last);
    if(channel_last){
        HMP_REQUIRE(tensor.stride(-2) == tensor.size(-1) && tensor.stride(-1) == 1,
           "cv::GpuMat require last two strides are contiguous, expect ({}, 1), got ({}, {})",
            tensor.size(-1), tensor.stride(-2), tensor.stride(-1));
    }
    else{
        HMP_REQUIRE(tensor.stride(-1) == 1, 
            "cv::GpuMat require last stride equal to 1, got {}", tensor.stride(-1));
    }

    int cn = channel_last ? tensor.size(-1) : 1;
    int type = to_cv_type(tensor.dtype(), cn);
    auto rows = tensor.size(0),
         cols = tensor.size(1);
    auto step = tensor.stride(0) * tensor.itemsize();
    return cv::cuda::GpuMat(rows, cols, type, tensor.unsafe_data(), step);
}


template<typename Func, typename ...Args>
void foreach_gpu_image(const Func &f, ChannelFormat cformat, Tensor &dst, Args&&...args)
{
    auto batch = dst.size(0);
    for(int64_t i = 0; i < batch; ++i){
        if(cformat == ChannelFormat::NCHW){
            for(int64_t c = 0; c < dst.size(1); ++c){
                auto dmat = to_cv_gpu_mat(dst.select(0, i).select(0, c));
                f(dmat, to_cv_gpu_mat(args.select(0, i).select(0, c))...);
            }
        }
        else{
            auto dmat = to_cv_gpu_mat(dst.select(0, i), dst.size(-1));
            f(dmat, to_cv_gpu_mat(args.select(0, i), args.size(-1))...);
        }
    }
}


static Tensor pitch_align_2d(const Tensor &t, bool copy=true, ChannelFormat cformat = ChannelFormat::NCHW)
{
    HMP_REQUIRE(cformat == ChannelFormat::NCHW, "pitch_align_2d only support NCHW layout");
    auto pitch_alignment = hmp::cuda::DeviceProp::texture_pitch_alignment();
    auto step = t.stride(-2) * t.itemsize(); //row stride
    if(step % pitch_alignment == 0){
        return t;
    }

    auto n = divup(t.size(-1), pitch_alignment/t.itemsize());
    auto cols = n * pitch_alignment / t.itemsize();
    auto alloc_shape = t.shape();
    alloc_shape[t.dim() - 1] = cols;
    auto tmp = empty(alloc_shape, t.options()).slice(-1, 0, t.size(-1));
    if(copy){
        hmp::copy(tmp, t);
    }
    return tmp;
}



static cv::cuda::Stream current_stream()
{
    return cv::cuda::StreamAccessor::wrapStream(
        hmp::kernel::cuda::getCurrentCUDAStream());
}

static void morph_cuda(cv::MorphTypes algo, const cv::cuda::GpuMat &src, cv::cuda::GpuMat &dst, cv::Mat &kernel)
{
    auto morph = cv::cuda::createMorphologyFilter(algo, src.type(), kernel);
    auto stream = current_stream();

    morph->apply(src, dst, stream);
}


}}} //namespace hmp::kernel::ocv