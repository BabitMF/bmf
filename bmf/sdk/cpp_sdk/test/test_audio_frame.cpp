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

#include <bmf/sdk/audio_frame.h>
#ifdef BMF_ENABLE_FFMPEG
#include <bmf/sdk/ffmpeg_helper.h>
#endif
#include <gtest/gtest.h>

using namespace bmf_sdk;

#ifdef BMF_ENABLE_FUZZTEST
#include <fuzztest/fuzztest.h>

using namespace fuzztest;

namespace { // helpers
void check_audio_frame_invariants(const AudioFrame &af, int samples, uint64_t layout, bool planar, ScalarType dtype) {
    const auto expected_num_channels = __builtin_popcountll(layout);
    EXPECT_TRUE(af);
    EXPECT_EQ(af.layout(), layout);
    EXPECT_EQ(af.dtype(), dtype);
    EXPECT_EQ(af.planer(), planar);
    EXPECT_EQ(af.nsamples(), samples);
    EXPECT_EQ(af.nchannels(), expected_num_channels);
    if (planar) { // expect a plane for each channel
        EXPECT_EQ(af.nplanes(), expected_num_channels);
        for (int i = 0; i < expected_num_channels; ++i) {
            EXPECT_NO_THROW(af[i]);
        }
        EXPECT_THROW(af[expected_num_channels], std::runtime_error);
    } else {
        EXPECT_EQ(af.nplanes(), 1); // interleaved
        EXPECT_NO_THROW(af[0]);
    }
    EXPECT_EQ(af.plane(0).size(0), samples);
    EXPECT_EQ(af.sample_rate(), 1); // default
}

auto AnyLayout() {
    return ElementOf<AudioChannelLayout::Layout>({
        AudioChannelLayout::kFRONT_LEFT,
        AudioChannelLayout::kFRONT_RIGHT,
        AudioChannelLayout::kFRONT_CENTER,
        AudioChannelLayout::kLOW_FREQUENCY,
        AudioChannelLayout::kBACK_LEFT,
        AudioChannelLayout::kBACK_RIGHT,
        AudioChannelLayout::kFRONT_LEFT_OF_CENTER,
        AudioChannelLayout::kFRONT_RIGHT_OF_CENTER,
        AudioChannelLayout::kBACK_CENTER,
        AudioChannelLayout::kSIDE_LEFT,
        AudioChannelLayout::kSIDE_RIGHT,
        AudioChannelLayout::kTOP_CENTER,
        AudioChannelLayout::kTOP_FRONT_LEFT,
        AudioChannelLayout::kTOP_FRONT_CENTER,
        AudioChannelLayout::kTOP_FRONT_RIGHT,
        AudioChannelLayout::kTOP_BACK_LEFT,
        AudioChannelLayout::kTOP_BACK_CENTER,
        AudioChannelLayout::kTOP_BACK_RIGHT,
        AudioChannelLayout::kSTEREO_LEFT,
        AudioChannelLayout::kSTEREO_RIGHT,
        AudioChannelLayout::kWIDE_LEFT,
        AudioChannelLayout::kWIDE_RIGHT,
        AudioChannelLayout::kSURROUND_DIRECT_LEFT,
        AudioChannelLayout::kSURROUND_DIRECT_RIGHT,
        AudioChannelLayout::kLOW_FREQUENCY_2,
        AudioChannelLayout::kLAYOUT_NATIVE,
        AudioChannelLayout::kLAYOUT_MONO,
        AudioChannelLayout::kLAYOUT_STEREO,
        AudioChannelLayout::kLAYOUT_2POINT1,
        AudioChannelLayout::kLAYOUT_2_1,
        AudioChannelLayout::kLAYOUT_SURROUND,
        AudioChannelLayout::kLAYOUT_3POINT1,
        AudioChannelLayout::kLAYOUT_4POINT0,
        AudioChannelLayout::kLAYOUT_4POINT1,
        AudioChannelLayout::kLAYOUT_2_2,
        AudioChannelLayout::kLAYOUT_QUAD,
        AudioChannelLayout::kLAYOUT_5POINT0,
        AudioChannelLayout::kLAYOUT_5POINT1,
        AudioChannelLayout::kLAYOUT_5POINT0_BACK,
        AudioChannelLayout::kLAYOUT_5POINT1_BACK,
        AudioChannelLayout::kLAYOUT_6POINT0,
        AudioChannelLayout::kLAYOUT_6POINT0_FRONT,
        AudioChannelLayout::kLAYOUT_HEXAGONAL,
        AudioChannelLayout::kLAYOUT_6POINT1,
        AudioChannelLayout::kLAYOUT_6POINT1_BACK,
        AudioChannelLayout::kLAYOUT_6POINT1_FRONT,
        AudioChannelLayout::kLAYOUT_7POINT0,
        AudioChannelLayout::kLAYOUT_7POINT0_FRONT,
        AudioChannelLayout::kLAYOUT_7POINT1,
        AudioChannelLayout::kLAYOUT_7POINT1_WIDE,
        AudioChannelLayout::kLAYOUT_7POINT1_WIDE_BACK,
        AudioChannelLayout::kLAYOUT_OCTAGONAL,
        AudioChannelLayout::kLAYOUT_HEXADECAGONAL,
        AudioChannelLayout::kLAYOUT_STEREO_DOWNMIX
    });
}

auto AnyDtype() {
    return ElementOf<ScalarType>({
#define ADD_ELEMENT(_, name) k##name, 
        HMP_FORALL_SCALAR_TYPES(ADD_ELEMENT)
#undef ADD_ELEMENT
    });
}

auto SamplesDomain() {
    return InRange<int64_t>(1, 16384);
}
} // namespace

void fuzz_constructor(int samples, uint64_t layout, bool planar, ScalarType dtype) {
    auto af = AudioFrame::make(samples, layout, planar, dtype);
    check_audio_frame_invariants(af, samples, layout, planar, dtype);
}

FUZZ_TEST(audio_frame, fuzz_constructor)
    .WithDomains(SamplesDomain(), AnyLayout(), Arbitrary<bool>(), AnyDtype());

void fuzz_constructor_w_tensor(SizeArray tensor_shape, uint64_t layout, bool planar, ScalarType dtype) {
    auto num_channels_external = __builtin_popcountll(layout); // external to the implementation under test
    auto num_dims = tensor_shape.size();

    if (planar) {
        TensorList tl{};
        for (int i = 0; i < num_channels_external; i++) {
            tl.emplace_back(empty(tensor_shape, dtype));
        }
        if (num_dims == 1) { // good planar construction
            auto af = AudioFrame::make(tl, layout, planar);
            check_audio_frame_invariants(af, tensor_shape[0], layout, planar, dtype);
        } else { // bad construction: expect throw
            EXPECT_THROW(AudioFrame::make(tl, layout, planar), std::runtime_error);
        }
    } else { // interleaved
        TensorList tl{empty(tensor_shape, dtype)};
        if (num_dims == 2 && tensor_shape[1] == num_channels_external) { // good interleaved construction
            auto af = AudioFrame::make(tl, layout, planar);
            check_audio_frame_invariants(af, tensor_shape[0], layout, planar, dtype);
        } else {
            EXPECT_THROW(AudioFrame::make(tl, layout, planar), std::runtime_error);
        }
    }
}

FUZZ_TEST(audio_frame, fuzz_constructor_w_tensor)
    .WithDomains(VectorOf(SamplesDomain()).WithMinSize(1).WithMaxSize(2), AnyLayout(), Arbitrary<bool>(), AnyDtype());
#endif // BMF_ENABLE_FUZZTEST

TEST(audio_frame, constructors) {
    AudioFrame af0;
    EXPECT_FALSE(af0);

    auto af1 = AudioFrame::make(8192, AudioChannelLayout::kLAYOUT_STEREO, false,
                                kInt16);
    EXPECT_TRUE(af1);
    EXPECT_EQ(af1.layout(), AudioChannelLayout::kLAYOUT_STEREO);
    EXPECT_EQ(af1.dtype(), kInt16);
    EXPECT_EQ(af1.planer(), false);
    EXPECT_EQ(af1.nsamples(), 8192);
    EXPECT_EQ(af1.nchannels(), 2);
    EXPECT_EQ(af1.sample_rate(), 1); // default
    EXPECT_EQ(af1.nplanes(), 1);     // interleave
    EXPECT_NO_THROW(af1[0]);
    EXPECT_THROW(af1[1], std::runtime_error);

    auto af2 = AudioFrame::make(8192, AudioChannelLayout::kLAYOUT_STEREO, true,
                                kInt16);
    EXPECT_TRUE(af2);
    EXPECT_TRUE(af2.planer());
    EXPECT_EQ(af2.nplanes(), 2);

    auto af3 = AudioFrame::make(8192, AudioChannelLayout::kLAYOUT_HEXADECAGONAL,
                                true, kInt16);
    EXPECT_TRUE(af3);
    EXPECT_TRUE(af3.planer());
    EXPECT_EQ(af3.nplanes(), 16);
    TensorList ad0;
    EXPECT_THROW(
        AudioFrame::make(ad0, AudioChannelLayout::kLAYOUT_STEREO, true),
        std::runtime_error);

    TensorList ad1{empty({8192, 2}, kInt16)}; // interleave stereo data
    EXPECT_THROW(
        AudioFrame::make(ad1, AudioChannelLayout::kLAYOUT_STEREO, true),
        std::runtime_error);
    EXPECT_NO_THROW(
        AudioFrame::make(ad1, AudioChannelLayout::kLAYOUT_STEREO, false));

    TensorList ad2{empty({8192}, kInt16),
                   empty({8192}, kInt16)}; // planer stereo data
    EXPECT_THROW(
        AudioFrame::make(ad2, AudioChannelLayout::kLAYOUT_STEREO, false),
        std::runtime_error);
    EXPECT_NO_THROW(
        AudioFrame::make(ad2, AudioChannelLayout::kLAYOUT_STEREO, true));
}

#ifdef BMF_ENABLE_FFMPEG
#ifdef BMF_ENABLE_FUZZTEST
void fuzz_ffmpeg_round_trip(int samples, uint64_t layout, bool planar, ScalarType dtype) {
    // construct fuzzy audio frame
    auto af0 = AudioFrame::make(samples, layout, planar, dtype);
    check_audio_frame_invariants(af0, samples, layout, planar, dtype);

    // convert to ffmpeg avframe
    AVFrame *aaf;
    try { // some formats may not be supported
        aaf = bmf_sdk::ffmpeg::from_audio_frame(af0, false);
    } catch (std::runtime_error) {
        return;
    }
    ASSERT_NE(aaf, nullptr);
    EXPECT_EQ(af0.planes()[0].tensorInfo().refcount(), 2); // af0, aaf
    for (int i = 0; i < FF_ARRAY_ELEMS(aaf->data); ++i) {
        EXPECT_EQ(aaf->data[i], aaf->extended_data[i]);
    }

    // convert back to audioframe
    AudioFrame af1;
    ASSERT_NO_THROW(af1 = bmf_sdk::ffmpeg::to_audio_frame(aaf, true));
    EXPECT_EQ(af0.planes()[0].tensorInfo().refcount(), 2); // private attach means the reference count wont increase
    av_frame_free(&aaf);
    EXPECT_EQ(af0.planes()[0].tensorInfo().refcount(), 2); // af0, af1
    check_audio_frame_invariants(af1, samples, layout, planar, dtype);
}

FUZZ_TEST(audio_frame, fuzz_ffmpeg_round_trip)
    .WithDomains(SamplesDomain(), AnyLayout(), Arbitrary<bool>(), AnyDtype());
#endif // BMF_ENABLE_FUZZTEST

TEST(audio_frame, ffmpeg_interop) {
    // planer
    {
        auto af0 = AudioFrame::make(8192, AudioChannelLayout::kLAYOUT_STEREO,
                                    true, kInt16);

        AVFrame *aaf0;
        ASSERT_NO_THROW(aaf0 = bmf_sdk::ffmpeg::from_audio_frame(af0, false));
        ASSERT_NE(aaf0, nullptr);

        EXPECT_EQ(aaf0->format, AV_SAMPLE_FMT_S16P);
        EXPECT_EQ(aaf0->channel_layout, AV_CH_LAYOUT_STEREO);
        EXPECT_EQ(aaf0->nb_samples, 8192);
        EXPECT_EQ(aaf0->channels, 2);
        EXPECT_EQ(aaf0->linesize[0], 8192 * sizeof(short));
        EXPECT_EQ(aaf0->nb_extended_buf, 0);
        for (int i = 0; i < FF_ARRAY_ELEMS(aaf0->data); ++i) {
            EXPECT_EQ(aaf0->data[i], aaf0->extended_data[i]);
        }
        //--> refcount check
        EXPECT_EQ(af0.planes()[0].tensorInfo().refcount(), 2); // af0, aaf0

        {
            AudioFrame af1;
            ASSERT_NO_THROW(af1 = bmf_sdk::ffmpeg::to_audio_frame(aaf0, true));
            EXPECT_EQ(af0.layout(), af1.layout());
            EXPECT_EQ(af0.dtype(), af1.dtype());
            EXPECT_EQ(af0.planer(), af1.planer());
            EXPECT_EQ(af0.nsamples(), af1.nsamples());
            EXPECT_EQ(af0.nchannels(), af1.nchannels());
            EXPECT_EQ(af0.nplanes(), af1.nplanes()); // interleave

            av_frame_free(&aaf0);
            EXPECT_EQ(af0.planes()[0].tensorInfo().refcount(), 2); // af0, af1
        }
        EXPECT_EQ(af0.planes()[0].tensorInfo().refcount(), 1); // af0
    }

    // interleave
    {
        auto af0 = AudioFrame::make(8192, AudioChannelLayout::kLAYOUT_STEREO,
                                    false, kInt16);

        AVFrame *aaf0;
        ASSERT_NO_THROW(aaf0 = bmf_sdk::ffmpeg::from_audio_frame(af0, false));
        ASSERT_NE(aaf0, nullptr);

        EXPECT_EQ(aaf0->format, AV_SAMPLE_FMT_S16);
        EXPECT_EQ(aaf0->channel_layout, AV_CH_LAYOUT_STEREO);
        EXPECT_EQ(aaf0->nb_samples, 8192);
        EXPECT_EQ(aaf0->channels, 2);
        EXPECT_EQ(aaf0->linesize[0], 8192 * sizeof(short) * 2);
        EXPECT_EQ(aaf0->nb_extended_buf, 0);
        for (int i = 0; i < FF_ARRAY_ELEMS(aaf0->data); ++i) {
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
            EXPECT_EQ(af0.nplanes(), af1.nplanes()); // interleave

            av_frame_free(&aaf0);
        }
    } //

    // channels > 8
    {
        auto af0 = AudioFrame::make(
            8192, AudioChannelLayout::kLAYOUT_HEXADECAGONAL, true, kInt16);

        AVFrame *aaf0;
        ASSERT_NO_THROW(aaf0 = bmf_sdk::ffmpeg::from_audio_frame(af0, false));
        ASSERT_NE(aaf0, nullptr);

        EXPECT_EQ(aaf0->format, AV_SAMPLE_FMT_S16P);
        EXPECT_EQ(aaf0->channel_layout, AV_CH_LAYOUT_HEXADECAGONAL);
        EXPECT_EQ(aaf0->nb_samples, 8192);
        EXPECT_EQ(aaf0->channels, 16);
        EXPECT_EQ(aaf0->linesize[0], 8192 * sizeof(short));
        EXPECT_EQ(aaf0->nb_extended_buf, 16 - FF_ARRAY_ELEMS(aaf0->buf));
        for (int i = 0; i < FF_ARRAY_ELEMS(aaf0->data); ++i) {
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
            EXPECT_EQ(af0.nplanes(), af1.nplanes()); // interleave

            av_frame_free(&aaf0);
        }
    } //
}
#endif

#ifdef BMF_ENABLE_FUZZTEST
void fuzz_copy_props(int samples, uint64_t layout, bool planar, ScalarType dtype, int time_base_denom, int pts, int sample_rate) {
    auto af0 = AudioFrame::make(samples, layout, planar, dtype);
    af0.set_time_base(Rational(1, time_base_denom));
    af0.set_pts(pts);
    af0.set_sample_rate(sample_rate);

    auto af1 = AudioFrame::make(samples, layout, planar, dtype);
    af1.copy_props(af0);

    EXPECT_EQ(af1.pts(), pts);
    EXPECT_EQ(af1.sample_rate(), sample_rate);
    EXPECT_EQ(af1.time_base().den, time_base_denom);
    EXPECT_EQ(af1.time_base().num, 1);
}

FUZZ_TEST(audio_frame, fuzz_copy_props)
    .WithDomains(SamplesDomain(), AnyLayout(), Arbitrary<bool>(), AnyDtype(), Positive<int>(), Positive<int>(), Positive<int>());
#endif // BMF_ENABLE_FUZZTEST

TEST(audio_frame, copy_props) {
    auto af0 = AudioFrame::make(8192, AudioChannelLayout::kLAYOUT_STEREO, false,
                                kInt16);
    af0.set_time_base(Rational(1, 2));
    af0.set_pts(100);
    af0.set_sample_rate(44100);

    auto af1 = AudioFrame::make(8192, AudioChannelLayout::kLAYOUT_STEREO, false,
                                kInt16);
    af1.copy_props(af0);

    EXPECT_EQ(af1.pts(), 100);
    EXPECT_EQ(af1.sample_rate(), 44100);
    EXPECT_EQ(af1.time_base().den, 2);
    EXPECT_EQ(af1.time_base().num, 1);
}