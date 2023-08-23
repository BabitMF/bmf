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
#include <bmf/sdk/config.h>
#ifdef BMF_ENABLE_TORCH
#include <bmf/sdk/torch_convertor.h>
#endif

#include <bmf/sdk/media_description.h>
#include <bmf/sdk/convert_backend.h>
#include <gtest/gtest.h>

using namespace bmf_sdk;

TEST(media_description, construct) {
    MediaDesc dp;
    dp.width(1920)
        .height(1080)
        .device(hmp::Device("cpu"))
        .pixel_format(hmp::PF_YUV420P)
        .media_type(MediaType::kAVFrame);

    EXPECT_EQ(dp.width(), 1920);
    EXPECT_EQ(dp.height(), 1080);
    EXPECT_EQ(dp.device().type(), hmp::Device::Type::CPU);
    EXPECT_EQ(dp.pixel_format(), hmp::PF_YUV420P);
    EXPECT_EQ(dp.media_type(), MediaType::kAVFrame);
}

TEST(media_description, has_value) {
    MediaDesc dp;
    dp.width(1920).device(hmp::Device("cpu")).pixel_format(hmp::PF_YUV420P);
    EXPECT_TRUE(dp.width.has_value());
    EXPECT_FALSE(dp.height.has_value());
    EXPECT_TRUE(dp.device.has_value());
    EXPECT_TRUE(dp.pixel_format.has_value());
    EXPECT_FALSE(dp.media_type.has_value());
}

TEST(convert_backend, format_cvt) {
    MediaDesc dp;
    dp.width(1920).pixel_format(hmp::PF_YUV420P);
    auto rgbformat = hmp::PixelInfo(hmp::PF_RGB24);
    auto src_vf = VideoFrame::make(640, 320, rgbformat);
    auto dst_vf = bmf_convert(src_vf, MediaDesc{}, dp);
    EXPECT_EQ(dst_vf.width(), 1920);
    EXPECT_EQ(dst_vf.height(), 960);
    EXPECT_EQ(dst_vf.frame().format(), hmp::PF_YUV420P);
}

#ifdef BMF_ENABLE_FFMPEG
// use for register
#include <bmf/sdk/av_convertor.h>
TEST(convert_backend, convert_AVFrame) {
    MediaDesc dp;
    dp.width(1920)
        .height(1080)
        .pixel_format(hmp::PF_RGB24)
        .media_type(MediaType::kAVFrame);
    auto yuvformat = hmp::PixelInfo(hmp::PF_YUV420P);
    auto src_vf = VideoFrame::make(640, 320, yuvformat);

    auto dst_vf = bmf_convert(src_vf, MediaDesc{}, dp);
    EXPECT_TRUE(static_cast<bool>(dst_vf));

    const AVFrame *frame = dst_vf.private_get<AVFrame>();
    EXPECT_EQ(frame->width, 1920);
    EXPECT_EQ(frame->height, 1080);
    EXPECT_EQ(frame->format, AV_PIX_FMT_RGB24);

    //
    VideoFrame src_with_avf;
    src_with_avf.private_attach<AVFrame>(frame);
    MediaDesc src_dp;
    src_dp.pixel_format(hmp::PF_RGB24).media_type(MediaType::kAVFrame);
    auto vf = bmf_convert(src_with_avf, src_dp, MediaDesc{});

    EXPECT_EQ(vf.width(), 1920);
    EXPECT_EQ(vf.height(), 1080);
    EXPECT_EQ(vf.frame().format(), hmp::PF_RGB24);
}

#endif

#ifdef HMP_ENABLE_OPENCV

#include <bmf/sdk/ocv_convertor.h>
TEST(convert_backend, convert_OCV) {
    MediaDesc dp;
    dp.width(1920)
        .height(1080)
        .pixel_format(hmp::PF_RGB24)
        .media_type(MediaType::kCVMat);
    auto yuvformat = hmp::PixelInfo(hmp::PF_YUV420P);
    auto src_vf = VideoFrame::make(640, 320, yuvformat);

    auto dst_vf = bmf_convert(src_vf, MediaDesc{}, dp);
    EXPECT_TRUE(static_cast<bool>(dst_vf));

    const cv::Mat *mat = dst_vf.private_get<cv::Mat>();
    EXPECT_TRUE(mat != NULL);

    mat->size();
    EXPECT_EQ(mat->dims, 2);
    EXPECT_EQ(mat->size[0], 1080);
    EXPECT_EQ(mat->size[1], 1920);

    cv::Mat *roi_mat = new cv::Mat((*mat)({0, 0, 1280, 720}));

    VideoFrame src_with_ocv;
    src_with_ocv.private_attach<cv::Mat>(roi_mat);
    MediaDesc src_dp;
    src_dp.pixel_format(hmp::PF_RGB24).media_type(MediaType::kCVMat);
    auto vf = bmf_convert(src_with_ocv, src_dp, MediaDesc{});
    EXPECT_EQ(vf.width(), 1280);
    EXPECT_EQ(vf.height(), 720);
    EXPECT_EQ(vf.frame().format(), hmp::PF_RGB24);

    cv::Mat *roi_mat2 = new cv::Mat((*mat)({0, 0, 1440, 720}));
    VideoFrame src_with_ocv2;
    src_with_ocv2.private_attach<cv::Mat>(roi_mat2);
    MediaDesc dst_dp;
    dst_dp.width(720).height(360).pixel_format(hmp::PF_YUV420P);
    auto vf2 = bmf_convert(src_with_ocv2, src_dp, dst_dp);
    EXPECT_EQ(vf2.width(), 720);
    EXPECT_EQ(vf2.height(), 360);
    EXPECT_EQ(vf2.frame().format(), hmp::PF_YUV420P);
}

#endif

#ifdef BMF_ENABLE_TORCH
TEST(convert_backend, convert_torch) {
    MediaDesc dp;
    dp.width(1920)
        .height(1080)
        .pixel_format(hmp::PF_RGB24)
        .media_type(MediaType::kATTensor);
    auto yuvformat = hmp::PixelInfo(hmp::PF_YUV420P);
    auto src_vf = VideoFrame::make(640, 320, yuvformat);

    auto dst_vf = bmf_convert(src_vf, MediaDesc{}, dp);
    EXPECT_TRUE(static_cast<bool>(dst_vf));

    const at::Tensor *att = dst_vf.private_get<at::Tensor>();
    EXPECT_TRUE(att != NULL);

    at::Tensor *att_new = new at::Tensor(*att);

    VideoFrame src_with_att;
    src_with_att.private_attach<at::Tensor>(att_new);
    MediaDesc src_dp;
    src_dp.pixel_format(hmp::PF_RGB24).media_type(MediaType::kATTensor);
    auto vf = bmf_convert(src_with_att, src_dp, MediaDesc{});
    EXPECT_EQ(vf.width(), 1920);
    EXPECT_EQ(vf.height(), 1080);
    EXPECT_EQ(vf.frame().format(), hmp::PF_RGB24);
}
#endif

#if defined(BMF_ENABLE_TORCH) && defined(BMF_ENABLE_CUDA)
TEST(convert_backend, convert_torch_cuda) {
    MediaDesc dp;
    dp.width(1920)
        .height(1080)
        .pixel_format(hmp::PF_RGB24)
        .device(hmp::Device("cuda"))
        .media_type(MediaType::kATTensor);
    auto yuvformat = hmp::PixelInfo(hmp::PF_YUV420P);
    auto src_vf = VideoFrame::make(640, 320, yuvformat);

    auto dst_vf = bmf_convert(src_vf, MediaDesc{}, dp);
    EXPECT_TRUE(static_cast<bool>(dst_vf));

    const at::Tensor *att = dst_vf.private_get<at::Tensor>();
    EXPECT_TRUE(att != NULL);

    at::Tensor *att_new = new at::Tensor(*att);

    VideoFrame src_with_att;
    src_with_att.private_attach<at::Tensor>(att_new);
    MediaDesc src_dp;
    src_dp.pixel_format(hmp::PF_RGB24).media_type(MediaType::kATTensor);
    auto vf = bmf_convert(src_with_att, src_dp, MediaDesc{});
    EXPECT_EQ(vf.width(), 1920);
    EXPECT_EQ(vf.height(), 1080);
    EXPECT_EQ(vf.frame().format(), hmp::PF_RGB24);
    EXPECT_EQ(vf.device().type(), hmp::Device::Type::CUDA);
}
#endif
