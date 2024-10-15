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

#ifdef BMF_ENABLE_FUZZTEST 
#include <fuzztest/fuzztest.h>

using namespace fuzztest;

namespace { // helper functions
auto AnyDtype() {
    return ElementOf<ScalarType>({
#define ADD_ELEMENT(_, name) k##name, 
        HMP_FORALL_SCALAR_TYPES(ADD_ELEMENT)
#undef ADD_ELEMENT
    });
}

auto AnyPixelFormat() {
    return ElementOf<PixelFormat>({
#define ADD_ELEMENT(name) hmp::name,
        HMP_FORALL_PIXEL_FORMATS(ADD_ELEMENT)
#undef ADD_ELEMENT
    });
}

auto AnyColorSpace() {
    return ElementOf<ColorSpace>({
#define ADD_ELEMENT(name) hmp::name,
        HMP_FORALL_COLOR_SPACES(ADD_ELEMENT)
#undef ADD_ELEMENT
    });
}

auto SizeRange() {
    return InRange(2, 7680); // max 8K
}

struct FuzzConvertParams {
    int width;
    int height;
    PixelFormat format;
    ColorSpace color_space;
};

auto FuzzConvertParamsDomain() {
    return StructOf<FuzzConvertParams>(
        SizeRange(),
        SizeRange(),
        AnyPixelFormat(),
        AnyColorSpace()
    );
}

void check_video_frame_invariants(const VideoFrame &vf, int width, int height, PixelFormat format, ColorSpace color_space) {
    ASSERT_EQ(vf.frame().format(), format);
    EXPECT_EQ(vf.height(), height);
    EXPECT_EQ(vf.width(), width);
    EXPECT_EQ(vf.frame().pix_info().space(), color_space);
}
} // namespace
#endif // BMF_ENABLE_FUZZTEST

namespace { // helper functions
void check_vf_equal(VideoFrame& vf1, VideoFrame& vf2) {
    EXPECT_EQ(vf1.dtype(), vf2.dtype());
    EXPECT_EQ(vf1.width(), vf2.width());
    EXPECT_EQ(vf1.height(), vf2.height());
    EXPECT_EQ(vf1.frame().format(), vf2.frame().format());
    EXPECT_EQ(vf1.frame().pix_info().space(), vf2.frame().pix_info().space());
    EXPECT_EQ(vf1.frame().pix_info().range(), vf2.frame().pix_info().range());
    EXPECT_EQ(vf1.device(), vf2.device());

    // check data equal
    for (int tensor_idx = 0; tensor_idx < vf1.frame().data().size(); ++tensor_idx) {
        auto t1 = vf1.frame().data()[tensor_idx];
        auto t2 = vf2.frame().data()[tensor_idx];
    }
}
} // namespace

TEST(media_description, construct) {
    MediaDesc dp;
    dp.width(1920)
        .height(1080)
        .device(hmp::Device("cpu"))
        .pixel_format(hmp::PF_YUV420P)
        .color_space(hmp::CS_BT709)
        .media_type(MediaType::kAVFrame);

    EXPECT_EQ(dp.width(), 1920);
    EXPECT_EQ(dp.height(), 1080);
    EXPECT_EQ(dp.device().type(), hmp::Device::Type::CPU);
    EXPECT_EQ(dp.pixel_format(), hmp::PF_YUV420P);
    EXPECT_EQ(dp.color_space(), hmp::CS_BT709);
    EXPECT_EQ(dp.media_type(), MediaType::kAVFrame);
}

TEST(media_description, construct_w_pix_info) {
    MediaDesc dp;
    dp.width(1920)
        .height(1080)
        .device(hmp::Device("cpu"))
        .pixel_info(PixelInfo(hmp::PF_YUV420P, hmp::ColorSpace::CS_BT709, hmp::CR_MPEG))
        .media_type(MediaType::kAVFrame);
    
    EXPECT_EQ(dp.width(), 1920);
    EXPECT_EQ(dp.height(), 1080);
    EXPECT_EQ(dp.device().type(), hmp::Device::Type::CPU);
    EXPECT_EQ(dp.pixel_info().format(), hmp::PF_YUV420P);
    EXPECT_EQ(dp.pixel_info().space(), hmp::CS_BT709);
    EXPECT_EQ(dp.pixel_info().range(), hmp::CR_MPEG);
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
    EXPECT_FALSE(dp.pixel_info.has_value());
}

#ifdef BMF_ENABLE_FUZZTEST
void fuzz_convert_round_trip(FuzzConvertParams src, FuzzConvertParams dst) {
    // construct videoframe
    auto src_pix_info = PixelInfo(src.format, src.color_space);
    auto src_vf = VideoFrame::make(src.width, src.height, src_pix_info);
    check_video_frame_invariants(src_vf, src.width, src.height, src.format, src.color_space);

    // convert to arbitrary type
    MediaDesc dp;
    dp.width(dst.width).height(dst.height).pixel_format(dst.format).color_space(dst.color_space);
    VideoFrame dst_vf;
    try {
        dst_vf = bmf_convert(src_vf, MediaDesc{}, dp);
    } catch (std::runtime_error&) {
        // an exception is expected if the conversion format is invalid (unsupported)
        return;
    }
    // since the conversion was successful, check invariants and and expect the return conversion to succeed
    check_video_frame_invariants(dst_vf, dst.width, dst.height, dst.format, dst.color_space);

    // return trip
    MediaDesc rp; 
    rp.width(src.width).height(src.height).pixel_info(src_pix_info);
    auto ret_vf = bmf_convert(dst_vf, MediaDesc{}, rp);
    check_video_frame_invariants(ret_vf, src.width, src.height, src.format, src.color_space);
    // TODO: do some validation on the underlying tensor data
}

FUZZ_TEST(convert_backend, fuzz_convert_round_trip)
    .WithDomains(FuzzConvertParamsDomain(), FuzzConvertParamsDomain());
#endif // BMF_ENABLE_FUZZTEST

TEST(convert_backend, format_cvt) {
    MediaDesc dp;
    dp.width(1920).pixel_format(hmp::PF_YUV420P);
    auto rgbformat = hmp::PixelInfo(hmp::PF_RGB24);
    auto src_vf = VideoFrame::make(640, 320, rgbformat);
    auto dst_vf = bmf_convert(src_vf, MediaDesc{}, dp);
    EXPECT_EQ(src_vf.width(), 640);
    EXPECT_EQ(dst_vf.width(), 1920);
    EXPECT_EQ(dst_vf.height(), 960);
    EXPECT_EQ(dst_vf.frame().format(), hmp::PF_YUV420P);
}

TEST(convert_backend, format_cvt_round_trip) {
    // depart - using MediaDesc::pix_info
    auto rgbformat = hmp::PixelInfo(hmp::PF_RGB24);
    auto yuvformat = hmp::PixelInfo(hmp::PF_YUV420P, hmp::CS_BT709, hmp::CR_MPEG);
    auto src_vf = VideoFrame::make(640, 320, rgbformat);
    MediaDesc tp;
    tp.width(1920).pixel_info(yuvformat);
    auto tmp_vf = bmf_convert(src_vf, MediaDesc{}, tp);
    EXPECT_EQ(src_vf.width(), 640);
    EXPECT_EQ(tmp_vf.width(), 1920);
    EXPECT_EQ(tmp_vf.height(), 960);
    EXPECT_EQ(tmp_vf.frame().format(), hmp::PF_YUV420P);
    EXPECT_EQ(tmp_vf.frame().pix_info().space(), hmp::CS_BT709);
    EXPECT_EQ(tmp_vf.frame().pix_info().range(), hmp::CR_MPEG);

    // return - using MediaDesc::pixel_format
    MediaDesc dp;
    dp.width(640)
        .height(320)
        .pixel_format(hmp::PF_RGB24);
    auto dst_vf = bmf_convert(tmp_vf, MediaDesc{}, dp);
    EXPECT_EQ(dst_vf.width(), 640);
    EXPECT_EQ(dst_vf.height(), 320);
    EXPECT_EQ(dst_vf.frame().format(), hmp::PF_RGB24);
    EXPECT_NE(dst_vf.frame().pix_info().space(), tmp_vf.frame().pix_info().space());
    EXPECT_NE(dst_vf.frame().pix_info().range(), tmp_vf.frame().pix_info().range());
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
