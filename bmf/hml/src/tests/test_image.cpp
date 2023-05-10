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
#include <gtest/gtest.h>
#include <hmp/tensor.h>
#include <hmp/imgproc.h>
#include <hmp/imgproc/image_seq.h>

using namespace hmp;

#ifdef HMP_ENABLE_FFMPEG

#include <hmp/ffmpeg/ffmpeg.h>

class ImageTest : public testing::Test
{
public:
    void SetUp() override
    {

    }

};


TEST_F(ImageTest, AVFrameInterOp)
{
#define BUF_REFCOUNT(t) t.tensorInfo()->buffer().refcount()
    auto pix_info = PixelInfo(PF_YUV420P, 
        ColorModel(CS_BT709, CR_MPEG, CP_BT709, CTC_BT709));

    auto f = Frame(1920, 1080, pix_info);

    for(int i = 0; i < f.nplanes(); ++i){
        f.plane(i).fill_((i+1)*3);
    }

    for(int i = 0; i < f.nplanes(); ++i){
        EXPECT_EQ(BUF_REFCOUNT(f.plane(i)), 1);
    }

    // Frame -> AVFrame -> Frame
    {
        auto vf = ffmpeg::to_video_frame(f);
        for(int i = 0; i < f.nplanes(); ++i){
            EXPECT_EQ(BUF_REFCOUNT(f.plane(i)), 2);
        }
        EXPECT_EQ(vf->colorspace, AVCOL_SPC_BT709);
        EXPECT_EQ(vf->color_range, AVCOL_RANGE_MPEG);
        EXPECT_EQ(vf->color_primaries, AVCOL_PRI_BT709);
        EXPECT_EQ(vf->color_trc, AVCOL_TRC_BT709);

        auto f1 = ffmpeg::from_video_frame(vf);
        av_frame_free(&vf); //f is still refed
        for(int i = 0; i < f.nplanes(); ++i){
            EXPECT_EQ(BUF_REFCOUNT(f.plane(i)), 2);
        }
    }
    //No ref to f now
    for(int i = 0; i < f.nplanes(); ++i){
        EXPECT_EQ(BUF_REFCOUNT(f.plane(i)), 1);
    }

    //Frame |-> AVFrame -> AVFrame -> Frame
    //      |-> AVFrame -> AVFrame -> Frame
    {
        auto vf0 = ffmpeg::to_video_frame(f);
        for(int i = 0; i < f.nplanes(); ++i){ //f is ref by vf0
            EXPECT_EQ(BUF_REFCOUNT(f.plane(i)), 2);
        }

        auto vf1 = ffmpeg::to_video_frame(f);
        for(int i = 0; i < f.nplanes(); ++i){ // f is ref by vf0, vf1
            EXPECT_EQ(BUF_REFCOUNT(f.plane(i)), 3);
        }

        auto vf2 = av_frame_clone(vf0); //
        av_frame_free(&vf0);
        for(int i = 0; i < f.nplanes(); ++i){ // f is ref by vf1, vf2
            EXPECT_EQ(BUF_REFCOUNT(f.plane(i)), 3);
        }

        auto vf3 = av_frame_clone(vf1); //
        av_frame_free(&vf1);
        for(int i = 0; i < f.nplanes(); ++i){ // f is ref by vf2, vf3
            EXPECT_EQ(BUF_REFCOUNT(f.plane(i)), 3);
        }

        auto f2 = ffmpeg::from_video_frame(vf2);
        av_frame_free(&vf2); //f is ref by f2, vf3
        for(int i = 0; i < f.nplanes(); ++i){
            EXPECT_EQ(BUF_REFCOUNT(f.plane(i)), 3);
        }

        auto f3 = ffmpeg::from_video_frame(vf3);
        av_frame_free(&vf3); //f is ref by f2, f3
        for(int i = 0; i < f.nplanes(); ++i){
            EXPECT_EQ(BUF_REFCOUNT(f.plane(i)), 3);
        }
        EXPECT_EQ(f3.format(), PF_YUV420P);
        EXPECT_EQ(f3.pix_info().space(), CS_BT709);
        EXPECT_EQ(f3.pix_info().range(), CR_MPEG);
        EXPECT_EQ(f3.pix_info().primaries(), CP_BT709);
        EXPECT_EQ(f3.pix_info().transfer_characteristic(), CTC_BT709);

        f2 = Frame(); //reset f2
        av_frame_free(&vf2); //f is ref by f3
        for(int i = 0; i < f.nplanes(); ++i){
            EXPECT_EQ(BUF_REFCOUNT(f.plane(i)), 2);
        }

    }
    //No ref to f now
    for(int i = 0; i < f.nplanes(); ++i){
        EXPECT_EQ(BUF_REFCOUNT(f.plane(i)), 1);
    }


#undef BUF_REFCOUNT

}
#endif //HMP_EANBLE_FFMPEG


TEST(tensor_op, hwc_to_chw)
{
    auto pix_info = PixelInfo(PF_RGB24);
    auto f = Frame(1920, 1080, pix_info);
    auto t = f.plane(0);
    EXPECT_EQ(t.dim(), 3);
    //H
    EXPECT_EQ(t.size(HWC_H), 1080);
    //W
    EXPECT_EQ(t.size(HWC_W), 1920);
    //C
    EXPECT_EQ(t.size(HWC_C), 3);

    auto nchw = img::transfer(t, kNHWC, kNCHW);
    EXPECT_EQ(nchw.dim(), 3);
    //C
    EXPECT_EQ(nchw.size(CHW_C), 3);
    //H
    EXPECT_EQ(nchw.size(CHW_H), 1080);
    //W
    EXPECT_EQ(nchw.size(CHW_W), 1920);
}

TEST(tensor_op, chw_to_hwc)
{
    //nchw
    auto pix_info = PixelInfo(PF_RGB24);
    auto f = Frame(1920, 1080, pix_info);
    auto t = f.plane(0);
    EXPECT_EQ(t.dim(), 3);
    //H
    EXPECT_EQ(t.size(HWC_H), 1080);
    //W
    EXPECT_EQ(t.size(HWC_W), 1920);
    //C
    EXPECT_EQ(t.size(HWC_C), 3);

    auto nchw = img::transfer(t, kNHWC, kNCHW);
    EXPECT_EQ(nchw.dim(), 3);
    //C
    EXPECT_EQ(nchw.size(CHW_C), 3);
    //H
    EXPECT_EQ(nchw.size(CHW_H), 1080);
    //W
    EXPECT_EQ(nchw.size(CHW_W), 1920);

    auto nhwc = img::transfer(nchw, kNCHW, kNHWC);
    EXPECT_EQ(nhwc.dim(), 3);
    //H
    EXPECT_EQ(nhwc.size(HWC_H), 1080);
    //W
    EXPECT_EQ(nhwc.size(HWC_W), 1920);
    //C
    EXPECT_EQ(nhwc.size(HWC_C), 3);
}

TEST(tensor_op, reformat_then_transfer)
{
    //nchw
    auto pix_info = PixelInfo(PF_YUV420P);
    auto f = Frame(1920, 1080, pix_info);
    auto rgb = PixelInfo(PF_RGB24);
    auto vf = f.reformat(rgb);

    auto t = vf.plane(0);
    EXPECT_EQ(t.dim(), 3);
    //H
    EXPECT_EQ(t.size(HWC_H), 1080);
    //W
    EXPECT_EQ(t.size(HWC_W), 1920);
    //C
    EXPECT_EQ(t.size(HWC_C), 3);

    auto nchw = img::transfer(t, kNHWC, kNCHW);
    EXPECT_EQ(nchw.dim(), 3);
    //C
    EXPECT_EQ(nchw.size(CHW_C), 3);
    //H
    EXPECT_EQ(nchw.size(CHW_H), 1080);
    //W
    EXPECT_EQ(nchw.size(CHW_W), 1920);
}

TEST(tensor_op, nhwc_to_nchw)
{
    auto pix_info = PixelInfo(PF_RGB24);
    auto f1 = Frame(1920, 1080, pix_info);
    auto f2 = Frame(1920, 1080, pix_info);
    FrameSeq fs = concat({f1, f2});

    auto t = fs.plane(0);

    //NHWC
    EXPECT_EQ(t.dim(), 4);
    //N
    EXPECT_EQ(t.size(NHWC_N), 2);
    //H
    EXPECT_EQ(t.size(NHWC_H), 1080);
    //W
    EXPECT_EQ(t.size(NHWC_W), 1920);
    //C
    EXPECT_EQ(t.size(NHWC_C), 3);

    auto nchw = img::transfer(t, kNHWC, kNCHW);
    EXPECT_EQ(nchw.dim(), 4);
    //N
    EXPECT_EQ(nchw.size(NCHW_N), 2);
    //C
    EXPECT_EQ(nchw.size(NCHW_C), 3);
    //H
    EXPECT_EQ(nchw.size(NCHW_H), 1080);
    //W
    EXPECT_EQ(nchw.size(NCHW_W), 1920);
}

TEST(tensor_op, nchw_to_nhwc)
{
    //nchw
    auto pix_info = PixelInfo(PF_RGB24);
    auto f1 = Frame(1920, 1080, pix_info);
    auto f2 = Frame(1920, 1080, pix_info);
    FrameSeq fs = concat({f1, f2});

    auto t = fs.plane(0);

    auto nchw = img::transfer(t, kNHWC, kNCHW);

    auto nhwc = img::transfer(nchw, kNCHW, kNHWC);

    //NHWC
    EXPECT_EQ(t.dim(), 4);
    //N
    EXPECT_EQ(nhwc.size(NHWC_N), 2);
    //H
    EXPECT_EQ(nhwc.size(NHWC_H), 1080);
    //W
    EXPECT_EQ(nhwc.size(NHWC_W), 1920);
    //C
    EXPECT_EQ(nhwc.size(NHWC_C), 3);

}



