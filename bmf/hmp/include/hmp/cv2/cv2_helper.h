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
#include <opencv2/core/mat.hpp>

namespace hmp {
namespace ocv {

#define HMP_FOR_ALL_CV_TYPES(_)                                                \
    _(CV_8U, kUInt8)                                                           \
    _(CV_8S, kInt8)                                                            \
    _(CV_16U, kUInt16)                                                         \
    _(CV_16S, kInt16)                                                          \
    _(CV_32S, kInt32)                                                          \
    _(CV_32F, kFloat32)                                                        \
    _(CV_64F, kFloat64)                                                        \
    _(CV_16F, kHalf)

inline std::tuple<ScalarType, int64_t> from_cv_type(int type) {
    auto depth = CV_MAT_DEPTH(type);
    int64_t cn = (type >> CV_CN_SHIFT) + 1;

    switch (depth) {
#define MAP(C, S)                                                              \
    case C:                                                                    \
        return std::make_tuple(S, cn);
        HMP_FOR_ALL_CV_TYPES(MAP)
#undef MAP
    default:
        HMP_REQUIRE(false, "unsupported OpenCV type {}", type);
    }
}

inline int to_cv_type(ScalarType dtype, int64_t cn = 0) {
    switch (dtype) {
#define MAP(C, S)                                                              \
    case S:                                                                    \
        return CV_MAKETYPE(C, cn);
        HMP_FOR_ALL_CV_TYPES(MAP)
#undef MAP
    default:
        HMP_REQUIRE(false, "{} is not supported by OpenCV", dtype);
    }
}

static Tensor from_cv_mat(const cv::Mat &mat) {
    ScalarType dtype;
    int64_t cn;
    std::tie(dtype, cn) = from_cv_type(mat.type());
    auto itemsize = sizeof_scalar_type(dtype);

    std::vector<int64_t> sizes;
    std::vector<int64_t> strides;
    for (int i = 0; i < mat.size.dims(); ++i) {
        sizes.push_back(mat.size[i]);
        if (i < mat.size.dims() - 1) {
            strides.push_back(mat.step[i] / itemsize);
        }
    }

    if (cn > 1) {
        sizes.push_back(cn);
        strides.push_back(cn);
    }
    strides.push_back(1);

    // no device index info in gpu::Mat
    return from_buffer(
        DataPtr(
            static_cast<void *>(mat.data), [mat](void *) {}, kCPU),
        dtype, sizes, strides);
}

static cv::Mat to_cv_mat(const Tensor &tensor, bool channel_last = true) {
    HMP_REQUIRE(tensor.device_type() == kCPU,
                "cv::Mat only support cpu tensor");
    HMP_REQUIRE(tensor.dim() - channel_last >= 2,
                "cv::Mat require dim >= 2, got {}",
                tensor.dim() - channel_last);
    if (channel_last) {
        HMP_REQUIRE(tensor.stride(-2) == tensor.size(-1) &&
                        tensor.stride(-1) == 1,
                    "cv::Mat require last two strides are contiguous, expect "
                    "({}, 1), got ({}, {})",
                    tensor.size(-1), tensor.stride(-2), tensor.stride(-1));
    } else {
        HMP_REQUIRE(tensor.stride(-1) == 1,
                    "cv::Mat require last stride equal to 1, got {}",
                    tensor.stride(-1));
    }

    int cn = channel_last ? tensor.size(-1) : 1;
    int type = to_cv_type(tensor.dtype(), cn);
    std::vector<int> sizes(tensor.shape().begin(),
                           tensor.shape().end() - channel_last);
    std::vector<size_t> steps;
    for (int64_t i = 0; i < tensor.dim() - 1 - channel_last; ++i) {
        steps.push_back(tensor.stride(i) * tensor.itemsize());
    }

    return cv::Mat(tensor.dim() - channel_last, sizes.data(), type,
                   tensor.unsafe_data(), steps.data());
}
} // namespace ocv
} // namespace hmp