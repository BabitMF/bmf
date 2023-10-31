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

#include <bmf/sdk/video_frame.h>
#include <bmf/sdk/json_param.h>
#include <bmf/sdk/module_functor.h>
#ifdef BMF_ENABLE_FFMPEG
#include <bmf/sdk/ffmpeg_helper.h>
#endif
#include <bmf/sdk/exception_factory.h>
#include <hmp/core/stream.h>
#include <gtest/gtest.h>

using namespace bmf_sdk;

// NOTE: all correctness of tensor operations are tested in hml/tests

template <typename T>
static bool check_pixel_value(const VideoFrame &vf, const T &v) {
    // ASSERT_TRUE(vf.is_image());
    // ASSERT_TRUE(vf.device() == kCPU);

    // RGB HWC
    auto frame = vf.frame();

    // only one plane
    auto &data = frame.plane(0);

    int64_t channels = data.size(hmp::HWC_C); //
    int64_t w = data.size(hmp::HWC_W);
    int64_t h = data.size(hmp::HWC_H);

    const T *ptr = data.data<T>();

    for (int64_t c = 0; c < channels; ++c) {
        for (int64_t y = 0; y < h; ++y) {
            for (int64_t x = 0; x < w; ++x) {
                auto idx = c * data.stride(hmp::HWC_C) +
                           y * data.stride(hmp::HWC_H) +
                           x * data.stride(hmp::HWC_W);
                if (ptr[idx] != v) {
                    return false;
                }
            }
        }
    }

    return true;
}

static VideoFrame decode_one_frame(const std::string &path) {
    JsonParam option;
    option.parse(fmt::format("{{\"input_path\": \"{}\"}}", path));
    auto decoder = make_sync_func<std::tuple<>, std::tuple<VideoFrame>>(
        ModuleInfo("c_ffmpeg_decoder"), option);

    VideoFrame vf;
    std::tie(vf) = decoder();
    return vf;
}

TEST(video_frame, frame_constructors) {
    int width = 1920, height = 1080;

    // default constructor
    VideoFrame undefined; //
    EXPECT_FALSE(undefined);

    // create with default TensorOptions
    auto H420 = PixelInfo(hmp::PF_YUV420P, hmp::CS_BT709);
    auto vf0 = VideoFrame::make(width, height, H420); //
    EXPECT_EQ(vf0.width(), width);
    EXPECT_EQ(vf0.height(), height);
    EXPECT_EQ(vf0.dtype(), kUInt8);
    EXPECT_TRUE(vf0.device() == kCPU);
    EXPECT_NO_THROW(vf0.frame());
    EXPECT_EQ(vf0.frame().format(), hmp::PF_YUV420P);

    // With dtype specified, not support

#ifdef HMP_ENABLE_CUDA
    // with device specifed
    auto vf2 = VideoFrame::make(1920, 1080, H420,
                                "cuda:0" // cuda device
    );
    EXPECT_TRUE(vf2.device() == Device(kCUDA, 0));
    EXPECT_TRUE(vf2.dtype() == kUInt8);

    auto vf3 =
        VideoFrame::make(1920, 1080, H420, Device(kCUDA, 0) // cuda device
        );
    EXPECT_TRUE(vf3.device() == Device(kCUDA, 0));

    auto vf4 = VideoFrame::make(1920, 1080, H420,
                                kCUDA // cuda device
    );
    EXPECT_TRUE(vf4.device() == Device(kCUDA, 0));

#endif
}

TEST(video_frame, crop_test) {
    int width = 1920, height = 1080;

    // frame
    auto H420 = PixelInfo(hmp::PF_YUV420P, hmp::CS_BT709);
    auto vf0 = VideoFrame::make(width, height, H420); //
    auto vf0_sub = vf0.crop(50, 100, 1280, 720);
    EXPECT_EQ(vf0_sub.width(), 1280);
    EXPECT_EQ(vf0_sub.height(), 720);
    auto &vf0_sub_data = vf0_sub.frame().plane(0);
    EXPECT_EQ(vf0_sub_data.stride(0),
              1920); //(H, W, 1) layout, stride(0) == line width
}

#ifdef HMP_ENABLE_CUDA

TEST(video_frame, copy_test) {
    int width = 1920, height = 1080;

    // frame
    auto H420 = PixelInfo(hmp::PF_YUV420P, hmp::CS_BT709);
    auto vf0 = VideoFrame::make(width, height, H420); //
    EXPECT_TRUE(vf0.device() == kCPU);
    auto vf0_cuda = vf0.cuda();
    EXPECT_TRUE(vf0_cuda.device() == kCUDA);
    auto vf0_cpu = vf0_cuda.cpu();
    EXPECT_TRUE(vf0_cpu.device() == kCPU);
}

TEST(video_frame, async_execution) {
    int width = 1920, height = 1080;
    auto RGB = PixelInfo(hmp::PF_RGB24, hmp::CS_BT709);
    auto vf0 = VideoFrame::make(width, height, RGB, kCPU); //
    auto vf1 = VideoFrame::make(width, height, RGB, kCUDA);

    VideoFrame vf2;
    //
    auto data = vf0.frame().data()[0]; // shadow copy, remove const
    data.fill_(1);                     //(1, 1, 1, 1, .....)

    // with cuda stream support
    auto stream = hmp::create_stream(kCUDA);
    {
        hmp::StreamGuard guard(stream);
        EXPECT_EQ(stream, hmp::current_stream(kCUDA));

        vf1.copy_(vf0);

        auto data = vf1.frame().data()[0];
        data += 2; // (3, 3, 3, 3, ....)

        vf2 = vf1.cpu(true); // async copy to cpu

        vf2.record();
        EXPECT_FALSE(vf2.ready());

        vf2.synchronize();
        // stream.synchronize();

        EXPECT_TRUE(vf2.ready());
    }

    ASSERT_EQ(vf2.dtype(), kUInt8);
    EXPECT_TRUE(check_pixel_value<uint8_t>(vf2, 3));
}

TEST(video_frame, from_hardware_avframe) {
    int width = 1920, height = 1080;

    auto NV12 = PixelInfo(hmp::PF_NV12, hmp::CS_BT709);
    AVFrame *avfrm =
        hmp::ffmpeg::hw_avframe_from_device(kCUDA, width, height, NV12);
    VideoFrame vf = ffmpeg::to_video_frame(avfrm);
    EXPECT_TRUE(vf.device() == kCUDA);
    auto from_video_frame = ffmpeg::from_video_frame(vf);
    EXPECT_TRUE(from_video_frame->hw_frames_ctx != NULL);
}

TEST(video_frame, gpu_frame_to_avframe) {
    int width = 1920, height = 1080;

    auto NV12 = PixelInfo(hmp::PF_NV12, hmp::CS_BT709);
    auto vf_gpu = VideoFrame::make(width, height, NV12, kCUDA);
    auto from_video_frame = ffmpeg::from_video_frame(vf_gpu, false);
    EXPECT_TRUE(from_video_frame->hw_frames_ctx != NULL);
}

TEST(video_frame, hardware_avframe_csc_resize) {
    int width = 1920, height = 1080;

    auto NV12 = PixelInfo(hmp::PF_NV12, hmp::CS_BT470BG);
    auto H420 = PixelInfo(hmp::PF_YUV420P, hmp::CS_BT709);
    // auto RGB24 = PixelInfo(hmp::PF_BGR24, hmp::CS_BT709);
    AVFrame *avfrm =
        hmp::ffmpeg::hw_avframe_from_device(kCUDA, width, height, NV12);
    VideoFrame vf = ffmpeg::to_video_frame(avfrm);
    EXPECT_TRUE(vf.device() == kCUDA);

    auto RGB = PixelInfo(hmp::PF_RGB24, hmp::CS_BT709);
    auto vf_rgb = VideoFrame::make(width, height, RGB, kCUDA);

    Tensor t_img = vf_rgb.frame().plane(0);
    hmp::img::yuv_to_rgb(t_img, vf.frame().data(), NV12, kNHWC);
    EXPECT_TRUE(t_img.device() == kCUDA);
    auto vf_yuv_from_rgb = VideoFrame::make(width, height, H420, kCUDA);
    TensorList tl = vf_yuv_from_rgb.frame().data();
    hmp::img::rgb_to_yuv(tl, vf_rgb.frame().data()[0], H420, kNHWC);

    auto vf_resize = VideoFrame::make(width / 2, height / 2, H420, kCUDA);
    // TensorList &yuv_resize(TensorList &dst, const TensorList &src,
    //                      PPixelFormat format, ImageFilterMode mode)
    // HMP_API TensorList &yuv_resize(TensorList &dst, const TensorList &src,
    // const PixelInfo &pix_info, ImageFilterMode mode =
    // ImageFilterMode::Bilinear);
    TensorList tl_resize = vf_resize.frame().data();
    hmp::img::yuv_resize(tl_resize, vf_yuv_from_rgb.frame().data(), H420);
    EXPECT_EQ(tl_resize.data()[0].size(0), height / 2);
    EXPECT_EQ(tl_resize.data()[0].size(1), width / 2);
}

#endif

namespace bmf_sdk {

struct MockAVFrame {
    MockAVFrame(bool *valid_p) : valid(valid_p) { *valid = true; }

    ~MockAVFrame() {
        if (valid) {
            *valid = false;
        }
    }

    int value = 0;

    bool *valid = nullptr;
};

template <> struct OpaqueDataInfo<MockAVFrame> {
    const static int key = OpaqueDataKey::kAVFrame;

    static OpaqueData construct(const MockAVFrame *avf) {
        return OpaqueData(const_cast<MockAVFrame *>(avf),
                          [](void *p) { delete (MockAVFrame *)p; });
    }
};

} // namespace bmf_sdk

TEST(video_frame, private_data) {
    int width = 1920, height = 1080;
    bool valid = false;

    {
        auto pri_data = new MockAVFrame(&valid);
        pri_data->value = 42;
        auto RGB = PixelInfo(hmp::PF_RGB24, hmp::CS_BT709);
        auto vf0 = VideoFrame::make(width, height, RGB); //
        vf0.private_attach(pri_data); // vf0 will own pri_data
        EXPECT_EQ(vf0.private_get<MockAVFrame>()->value, 42);
        EXPECT_EQ(valid, true);

        //
        auto vf1 = VideoFrame::make(width, height, RGB); //
        vf1.copy_(vf0);
        vf1.private_merge(
            vf0); // now, vf0 and vf1 will share the same private data
        EXPECT_EQ(vf1.private_get<MockAVFrame>()->value, 42);

        // vf1.private_get<MockAVFrame>()->value = 100; //modify already set
        // private data is not allowed
        // as it may cause unpredictable error(it may used in other modules)
        // solution to modify private data is to copy it
        auto pri_data_copy =
            new MockAVFrame(*vf0.private_get<MockAVFrame>()); // copy
        EXPECT_EQ(pri_data_copy->value, 42);
        pri_data_copy->valid = nullptr;
        pri_data_copy->value = 100;
        vf1.private_attach(pri_data_copy);
        EXPECT_EQ(vf1.private_get<MockAVFrame>()->value, 100);
        EXPECT_EQ(vf0.private_get<MockAVFrame>()->value, 42);

        //
        EXPECT_TRUE(valid); // ensure pri_data is alive
    }

    EXPECT_FALSE(valid); // pri_data is destructed
}

TEST(video_frame, private_data_json_param) {
    auto RGB = PixelInfo(hmp::PF_RGB24, hmp::CS_BT709);
    auto vf = VideoFrame::make(1920, 1080, RGB); //
    auto json_sptr = vf.private_get<JsonParam>();
    EXPECT_FALSE(json_sptr);

    JsonParam ref;
    ref.parse("{\"v\": 42}");
    vf.private_attach(&ref); // copy it internally

    auto data_sptr = vf.private_get<JsonParam>();
    ASSERT_TRUE(data_sptr);

    EXPECT_EQ(data_sptr->get<int>("v"), 42);
}

TEST(video_frame, copy_props) {
    auto H420 = PixelInfo(hmp::PF_YUV420P, hmp::CS_BT709);
    int width = 1920, height = 1080;
    auto RGB = PixelInfo(hmp::PF_RGB24, hmp::CS_BT709);
    auto vf0 = VideoFrame::make(width, height, RGB);  // Image type
    auto vf1 = VideoFrame::make(width, height, H420); // Frame type

    vf0.set_stream(42);
    vf0.set_time_base(Rational(1, 2));
    vf0.set_pts(100);

    vf1.copy_props(vf0);
    EXPECT_EQ(vf1.stream(), 42);
    EXPECT_EQ(vf1.pts(), 100);
    EXPECT_EQ(vf1.time_base().den, 2);
    EXPECT_EQ(vf1.time_base().num, 1);
}

TEST(video_frame, reformat) {
    auto ori_vf = decode_one_frame("../../files/big_bunny_10s_30fps.mp4");
    ASSERT_EQ(ori_vf.frame().format(), hmp::PF_YUV420P);
    EXPECT_EQ(ori_vf.height(), 1080);
    EXPECT_EQ(ori_vf.width(), 1920);
    ASSERT_FALSE(ori_vf.frame().pix_info().is_rgbx());
    EXPECT_EQ(ori_vf.frame().nplanes(), 3);
    EXPECT_EQ(ori_vf.frame().plane(0).stride(0), 1920);
    EXPECT_EQ(ori_vf.frame().plane(1).stride(0), 1920 / 2);
    EXPECT_EQ(ori_vf.frame().plane(2).stride(0), 1920 / 2);

    // reformat yuv420p -> rgb
    {
        auto RGB = PixelInfo(hmp::PF_RGB24, hmp::CS_BT709);
        auto rgb_vf = ori_vf.reformat(RGB);
        EXPECT_EQ(rgb_vf.height(), 1080);
        EXPECT_EQ(rgb_vf.width(), 1920);
        EXPECT_EQ(rgb_vf.frame().nplanes(), 1);
        EXPECT_EQ(rgb_vf.frame().height(), 1080);
        EXPECT_EQ(rgb_vf.frame().width(), 1920);
        ASSERT_EQ(rgb_vf.frame().format(), hmp::PF_RGB24);
        ASSERT_TRUE(rgb_vf.frame().pix_info().is_rgbx());
        EXPECT_EQ(rgb_vf.frame().plane(0).stride(0), 3 * 1920);
    }

    auto RGB = PixelInfo(hmp::PF_RGB24, hmp::CS_BT709);

    {
        auto img_vf = ori_vf.reformat(RGB);
        auto H420 = PixelInfo(hmp::PF_YUV420P, hmp::CS_BT709);
        auto yuv_vf = img_vf.reformat(H420);
        EXPECT_EQ(yuv_vf.height(), 1080);
        EXPECT_EQ(yuv_vf.width(), 1920);
        EXPECT_EQ(yuv_vf.frame().nplanes(), 3);
        EXPECT_EQ(yuv_vf.frame().height(), 1080);
        EXPECT_EQ(yuv_vf.frame().width(), 1920);
        ASSERT_EQ(yuv_vf.frame().format(), hmp::PF_YUV420P);
        ASSERT_FALSE(yuv_vf.frame().pix_info().is_rgbx());
        EXPECT_EQ(yuv_vf.frame().plane(0).stride(0), 1920);
        EXPECT_EQ(yuv_vf.frame().plane(1).stride(0), 1920 / 2);
        EXPECT_EQ(yuv_vf.frame().plane(2).stride(0), 1920 / 2);
    }

    {
        auto img_vf = ori_vf.reformat(RGB);
        auto NV12_709 = PixelInfo(hmp::PF_NV12, hmp::CS_BT709);
        auto yuv_vf = img_vf.reformat(NV12_709);
        EXPECT_EQ(yuv_vf.height(), 1080);
        EXPECT_EQ(yuv_vf.width(), 1920);
        EXPECT_EQ(yuv_vf.frame().nplanes(), 2);
        EXPECT_EQ(yuv_vf.frame().height(), 1080);
        EXPECT_EQ(yuv_vf.frame().width(), 1920);
        ASSERT_EQ(yuv_vf.frame().format(), hmp::PF_NV12);
        ASSERT_FALSE(yuv_vf.frame().pix_info().is_rgbx());
        EXPECT_EQ(yuv_vf.frame().plane(0).stride(0), 1920);
        EXPECT_EQ(yuv_vf.frame().plane(1).size(0), 1080 / 2);
        EXPECT_EQ(yuv_vf.frame().plane(1).size(1), 1920 / 2);
        EXPECT_EQ(yuv_vf.frame().plane(1).size(2), 2);

        //
        auto new_img = yuv_vf.reformat(RGB);
        EXPECT_EQ(new_img.height(), 1080);
        EXPECT_EQ(new_img.width(), 1920);
        EXPECT_EQ(new_img.frame().nplanes(), 1);
        EXPECT_EQ(new_img.frame().height(), 1080);
        EXPECT_EQ(new_img.frame().width(), 1920);
        ASSERT_EQ(new_img.frame().format(), hmp::PF_RGB24);
        ASSERT_TRUE(new_img.frame().pix_info().is_rgbx());
        EXPECT_EQ(new_img.frame().plane(0).stride(0), 3 * 1920);
    }
}

#ifdef BMF_ENABLE_FFMPEG
TEST(video_frame, reformat_by_ffmpeg) {
    auto ori_vf = decode_one_frame("../../files/big_bunny_10s_30fps.mp4");
    ASSERT_EQ(ori_vf.frame().format(), hmp::PF_YUV420P);
    EXPECT_EQ(ori_vf.height(), 1080);
    EXPECT_EQ(ori_vf.width(), 1920);
    ASSERT_FALSE(ori_vf.frame().pix_info().is_rgbx());
    EXPECT_EQ(ori_vf.frame().nplanes(), 3);
    EXPECT_EQ(ori_vf.frame().plane(0).stride(0), 1920);
    EXPECT_EQ(ori_vf.frame().plane(1).stride(0), 1920 / 2);
    EXPECT_EQ(ori_vf.frame().plane(2).stride(0), 1920 / 2);

    // reformat yuv420p -> rgb
    {
        auto rgb_vf = ffmpeg::reformat(ori_vf, "rgb24");
        EXPECT_EQ(rgb_vf.height(), 1080);
        EXPECT_EQ(rgb_vf.width(), 1920);
        EXPECT_EQ(rgb_vf.frame().nplanes(), 1);
        EXPECT_EQ(rgb_vf.frame().height(), 1080);
        EXPECT_EQ(rgb_vf.frame().width(), 1920);
        ASSERT_EQ(rgb_vf.frame().format(), hmp::PF_RGB24);
        ASSERT_TRUE(rgb_vf.frame().pix_info().is_rgbx());
        EXPECT_EQ(rgb_vf.frame().plane(0).stride(0), 3 * 1920);
    }

    auto img_vf = ffmpeg::reformat(ori_vf, "rgb24");

    {
        auto gray_vf = ffmpeg::reformat(img_vf, "gray");
        EXPECT_EQ(gray_vf.height(), 1080);
        EXPECT_EQ(gray_vf.width(), 1920);
        EXPECT_EQ(gray_vf.frame().nplanes(), 1);
        EXPECT_EQ(gray_vf.frame().height(), 1080);
        EXPECT_EQ(gray_vf.frame().width(), 1920);
        ASSERT_EQ(gray_vf.frame().format(), hmp::PF_GRAY8);
        ASSERT_TRUE(gray_vf.frame().pix_info().is_rgbx());
        EXPECT_EQ(gray_vf.frame().plane(0).stride(0), 1920);
    }

    // Image reformat
    {
        auto gray_vf = ffmpeg::reformat(img_vf, "gray16");
        EXPECT_EQ(gray_vf.height(), 1080);
        EXPECT_EQ(gray_vf.width(), 1920);
        EXPECT_EQ(gray_vf.frame().nplanes(), 1);
        EXPECT_EQ(gray_vf.frame().height(), 1080);
        EXPECT_EQ(gray_vf.frame().width(), 1920);
        ASSERT_EQ(gray_vf.frame().format(), hmp::PF_GRAY16);
        ASSERT_TRUE(gray_vf.frame().pix_info().is_rgbx());
        EXPECT_EQ(gray_vf.frame().plane(0).stride(0), 1920);
    }

    {
        auto rgb_vf = ffmpeg::reformat(img_vf, "rgb24");
        EXPECT_EQ(rgb_vf.height(), 1080);
        EXPECT_EQ(rgb_vf.width(), 1920);
        EXPECT_EQ(rgb_vf.frame().nplanes(), 1);
        EXPECT_EQ(rgb_vf.frame().height(), 1080);
        EXPECT_EQ(rgb_vf.frame().width(), 1920);
        ASSERT_EQ(rgb_vf.frame().format(), hmp::PF_RGB24);
        ASSERT_TRUE(rgb_vf.frame().pix_info().is_rgbx());
        EXPECT_EQ(rgb_vf.frame().plane(0).stride(0), 3 * 1920);
    }

    {
        auto yuv_vf = ffmpeg::reformat(img_vf, "yuv420p");
        EXPECT_EQ(yuv_vf.height(), 1080);
        EXPECT_EQ(yuv_vf.width(), 1920);
        EXPECT_EQ(yuv_vf.frame().nplanes(), 3);
        EXPECT_EQ(yuv_vf.frame().height(), 1080);
        EXPECT_EQ(yuv_vf.frame().width(), 1920);
        ASSERT_EQ(yuv_vf.frame().format(), hmp::PF_YUV420P);
        ASSERT_FALSE(yuv_vf.frame().pix_info().is_rgbx());
        EXPECT_EQ(yuv_vf.frame().plane(0).stride(0), 1920);
        EXPECT_EQ(yuv_vf.frame().plane(1).stride(0), 1920 / 2);
        EXPECT_EQ(yuv_vf.frame().plane(2).stride(0), 1920 / 2);
    }

    // Frame reformat
    {
        auto yuv_vf = ffmpeg::reformat(ori_vf, "yuv420p");
        EXPECT_EQ(yuv_vf.height(), 1080);
        EXPECT_EQ(yuv_vf.width(), 1920);
        EXPECT_EQ(yuv_vf.frame().nplanes(), 3);
        EXPECT_EQ(yuv_vf.frame().height(), 1080);
        EXPECT_EQ(yuv_vf.frame().width(), 1920);
        ASSERT_EQ(yuv_vf.frame().format(), hmp::PF_YUV420P);
        ASSERT_FALSE(yuv_vf.frame().pix_info().is_rgbx());
        EXPECT_EQ(yuv_vf.frame().plane(0).stride(0), 1920);
        EXPECT_EQ(yuv_vf.frame().plane(1).stride(0), 1920 / 2);
        EXPECT_EQ(yuv_vf.frame().plane(2).stride(0), 1920 / 2);
    }
}

#ifdef HMP_ENABLE_CUDA
TEST(video_frame, check_continue) {
    int width = 1920;
    int height = 1080;
    auto H420 = PixelInfo(hmp::PF_YUV420P, hmp::CS_BT709);
    auto vf = VideoFrame::make(width, height, H420, Device(kCUDA, 0)); //
    EXPECT_EQ(hmp::ffmpeg::check_frame_memory_is_contiguous(vf.frame()),
              false);
}

TEST(video_frame, copy_frame_test) {
    static const int count = 100;
    VideoFrame vf[count];
    int width = 1920;
    int height = 1080;
    auto H420 = PixelInfo(hmp::PF_YUV420P, hmp::CS_BT709);
    for (int i = 0; i < count; ++i) {
        vf[i] = VideoFrame::make(width, height, H420); //
    }

    {
        for (int i = 0; i < count; ++i) {
            auto cudaframe = vf[i].cuda();
            EXPECT_EQ(hmp::ffmpeg::check_frame_memory_is_contiguous(
                          cudaframe.frame()),
                      false);
        }
    }
}
#endif

#endif

TEST(video_frame, yuv_frame_storage) {
    int width = 1920;
    int height = 1080;
    auto H420 = PixelInfo(hmp::PF_YUV420P, hmp::CS_BT709);
    auto vf = VideoFrame::make(width, height, H420); //
    ASSERT_TRUE(vf.frame().storage().defined());
}
