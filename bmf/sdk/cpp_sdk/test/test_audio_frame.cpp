
#include <bmf/sdk/audio_frame.h>
#ifdef BMF_ENABLE_FFMPEG
#include <bmf/sdk/ffmpeg_helper.h>
#endif
#include <gtest/gtest.h>


using namespace bmf_sdk;


TEST(audio_frame, constructors)
{
    AudioFrame af0;
    EXPECT_FALSE(af0);

    auto af1 = AudioFrame::make(8192, AudioChannelLayout::kLAYOUT_STEREO, false, kInt16);
    EXPECT_TRUE(af1);   
    EXPECT_EQ(af1.layout(), AudioChannelLayout::kLAYOUT_STEREO);
    EXPECT_EQ(af1.dtype(), kInt16);
    EXPECT_EQ(af1.planer(), false);
    EXPECT_EQ(af1.nsamples(), 8192);
    EXPECT_EQ(af1.nchannels(), 2);
    EXPECT_EQ(af1.sample_rate(), 1); //default
    EXPECT_EQ(af1.nplanes(), 1); //interleave
    EXPECT_NO_THROW(af1[0]);
    EXPECT_THROW(af1[1], std::runtime_error);

    auto af2 = AudioFrame::make(8192, AudioChannelLayout::kLAYOUT_STEREO, true, kInt16);
    EXPECT_TRUE(af2);
    EXPECT_TRUE(af2.planer());
    EXPECT_EQ(af2.nplanes(), 2);

    auto af3 = AudioFrame::make(8192, AudioChannelLayout::kLAYOUT_HEXADECAGONAL, true, kInt16);
    EXPECT_TRUE(af3);
    EXPECT_TRUE(af3.planer());
    EXPECT_EQ(af3.nplanes(), 16);

    TensorList ad0;
    EXPECT_THROW(AudioFrame::make(ad0, AudioChannelLayout::kLAYOUT_STEREO, true),
                 std::runtime_error);

    TensorList ad1{empty({8192, 2}, kInt16)}; //interleave stereo data
    EXPECT_THROW(AudioFrame::make(ad1, AudioChannelLayout::kLAYOUT_STEREO, true),
                 std::runtime_error);
    EXPECT_NO_THROW(AudioFrame::make(ad1, AudioChannelLayout::kLAYOUT_STEREO, false));


    TensorList ad2{empty({8192}, kInt16), empty({8192}, kInt16)}; //planer stereo data
    EXPECT_THROW(AudioFrame::make(ad2, AudioChannelLayout::kLAYOUT_STEREO, false),
                 std::runtime_error);
    EXPECT_NO_THROW(AudioFrame::make(ad2, AudioChannelLayout::kLAYOUT_STEREO, true));
}

#ifdef BMF_ENABLE_FFMPEG
TEST(audio_frame, ffmpeg_interop)
{
    //planer
    {
        auto af0 = AudioFrame::make(8192, AudioChannelLayout::kLAYOUT_STEREO, true, kInt16);

        AVFrame* aaf0;
        ASSERT_NO_THROW(aaf0 = bmf_sdk::ffmpeg::from_audio_frame(af0, false));
        ASSERT_NE(aaf0, nullptr);

        EXPECT_EQ(aaf0->format, AV_SAMPLE_FMT_S16P);
        EXPECT_EQ(aaf0->channel_layout, AV_CH_LAYOUT_STEREO);
        EXPECT_EQ(aaf0->nb_samples, 8192);
        EXPECT_EQ(aaf0->channels, 2);
        EXPECT_EQ(aaf0->linesize[0], 8192*sizeof(short));
        EXPECT_EQ(aaf0->nb_extended_buf, 0);
        for(int i = 0; i < FF_ARRAY_ELEMS(aaf0->data); ++i){
            EXPECT_EQ(aaf0->data[i], aaf0->extended_data[i]);
        }
        //--> refcount check
        EXPECT_EQ(af0.planes()[0].tensorInfo().refcount(), 2); //af0, aaf0

        {
            AudioFrame af1;
            ASSERT_NO_THROW(af1 = bmf_sdk::ffmpeg::to_audio_frame(aaf0, true));
            EXPECT_EQ(af0.layout(), af1.layout());
            EXPECT_EQ(af0.dtype(), af1.dtype());
            EXPECT_EQ(af0.planer(), af1.planer());
            EXPECT_EQ(af0.nsamples(), af1.nsamples());
            EXPECT_EQ(af0.nchannels(), af1.nchannels());
            EXPECT_EQ(af0.nplanes(), af1.nplanes()); //interleave

            av_frame_free(&aaf0);
            EXPECT_EQ(af0.planes()[0].tensorInfo().refcount(), 2); //af0, af1
        }
        EXPECT_EQ(af0.planes()[0].tensorInfo().refcount(), 1); //af0
    }

    //interleave
    {
        auto af0 = AudioFrame::make(8192, AudioChannelLayout::kLAYOUT_STEREO, false, kInt16);

        AVFrame* aaf0;
        ASSERT_NO_THROW(aaf0 = bmf_sdk::ffmpeg::from_audio_frame(af0, false));
        ASSERT_NE(aaf0, nullptr);

        EXPECT_EQ(aaf0->format, AV_SAMPLE_FMT_S16);
        EXPECT_EQ(aaf0->channel_layout, AV_CH_LAYOUT_STEREO);
        EXPECT_EQ(aaf0->nb_samples, 8192);
        EXPECT_EQ(aaf0->channels, 2);
        EXPECT_EQ(aaf0->linesize[0], 8192*sizeof(short)*2);
        EXPECT_EQ(aaf0->nb_extended_buf, 0);
        for(int i = 0; i < FF_ARRAY_ELEMS(aaf0->data); ++i){
            EXPECT_EQ(aaf0->data[i], aaf0->extended_data[i]);
        }

        {
            AudioFrame af1;
            ASSERT_NO_THROW(af1 = bmf_sdk::ffmpeg::to_audio_frame(aaf0, true));
            EXPECT_EQ(af0.layout(), af1.layout());
            EXPECT_EQ(af0.dtype(), af1.dtype());
            EXPECT_EQ(af0.planer(), af1.planer());
            EXPECT_EQ(af0.nsamples(), af1.nsamples());
            EXPECT_EQ(af0.nchannels(), af1.nchannels());
            EXPECT_EQ(af0.nplanes(), af1.nplanes()); //interleave

            av_frame_free(&aaf0);
        }
    } //

    //channels > 8
    {
        auto af0 = AudioFrame::make(8192, AudioChannelLayout::kLAYOUT_HEXADECAGONAL, true, kInt16);

        AVFrame* aaf0;
        ASSERT_NO_THROW(aaf0 = bmf_sdk::ffmpeg::from_audio_frame(af0, false));
        ASSERT_NE(aaf0, nullptr);

        EXPECT_EQ(aaf0->format, AV_SAMPLE_FMT_S16P);
        EXPECT_EQ(aaf0->channel_layout, AV_CH_LAYOUT_HEXADECAGONAL);
        EXPECT_EQ(aaf0->nb_samples, 8192);
        EXPECT_EQ(aaf0->channels, 16);
        EXPECT_EQ(aaf0->linesize[0], 8192*sizeof(short));
        EXPECT_EQ(aaf0->nb_extended_buf, 16-FF_ARRAY_ELEMS(aaf0->buf));
        for(int i = 0; i < FF_ARRAY_ELEMS(aaf0->data); ++i){
            EXPECT_EQ(aaf0->data[i], aaf0->extended_data[i]);
        }

        {
            AudioFrame af1;
            ASSERT_NO_THROW(af1 = bmf_sdk::ffmpeg::to_audio_frame(aaf0, true));
            EXPECT_EQ(af0.layout(), af1.layout());
            EXPECT_EQ(af0.dtype(), af1.dtype());
            EXPECT_EQ(af0.planer(), af1.planer());
            EXPECT_EQ(af0.nsamples(), af1.nsamples());
            EXPECT_EQ(af0.nchannels(), af1.nchannels());
            EXPECT_EQ(af0.nplanes(), af1.nplanes()); //interleave

            av_frame_free(&aaf0);
        }
    } //


}
#endif



TEST(audio_frame, copy_props)
{
    auto af0 = AudioFrame::make(8192, AudioChannelLayout::kLAYOUT_STEREO, false, kInt16);
    af0.set_time_base(Rational(1, 2));
    af0.set_pts(100);
    af0.set_sample_rate(44100);

    auto af1 = AudioFrame::make(8192, AudioChannelLayout::kLAYOUT_STEREO, false, kInt16);
    af1.copy_props(af0);

    EXPECT_EQ(af1.pts(), 100);
    EXPECT_EQ(af1.sample_rate(), 44100);
    EXPECT_EQ(af1.time_base().den, 2);
    EXPECT_EQ(af1.time_base().num, 1);
}