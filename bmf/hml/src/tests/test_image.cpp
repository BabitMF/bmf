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

#ifdef HMP_ENABLE_FFMPEG

#include <hmp/ffmpeg/ffmpeg.h>

using namespace hmp;

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