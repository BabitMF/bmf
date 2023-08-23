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

#include <bmf/sdk/hmp_import.h>
#include <bmf/sdk/sdk_interface.h>
#include <bmf/sdk/bmf_type_info.h>

namespace bmf_sdk {

struct AudioChannelLayout {
    // ref: ffmpeg/channel_layout.h
    enum Layout {
        kFRONT_LEFT = 0x00000001,
        kFRONT_RIGHT = 0x00000002,
        kFRONT_CENTER = 0x00000004,
        kLOW_FREQUENCY = 0x00000008,
        kBACK_LEFT = 0x00000010,
        kBACK_RIGHT = 0x00000020,
        kFRONT_LEFT_OF_CENTER = 0x00000040,
        kFRONT_RIGHT_OF_CENTER = 0x00000080,
        kBACK_CENTER = 0x00000100,
        kSIDE_LEFT = 0x00000200,
        kSIDE_RIGHT = 0x00000400,
        kTOP_CENTER = 0x00000800,
        kTOP_FRONT_LEFT = 0x00001000,
        kTOP_FRONT_CENTER = 0x00002000,
        kTOP_FRONT_RIGHT = 0x00004000,
        kTOP_BACK_LEFT = 0x00008000,
        kTOP_BACK_CENTER = 0x00010000,
        kTOP_BACK_RIGHT = 0x00020000,
        kSTEREO_LEFT = 0x20000000,
        kSTEREO_RIGHT = 0x40000000,
        kWIDE_LEFT = 0x0000000080000000ULL,
        kWIDE_RIGHT = 0x0000000100000000ULL,
        kSURROUND_DIRECT_LEFT = 0x0000000200000000ULL,
        kSURROUND_DIRECT_RIGHT = 0x0000000400000000ULL,
        kLOW_FREQUENCY_2 = 0x0000000800000000ULL,

        kLAYOUT_NATIVE = 0x8000000000000000ULL,

        kLAYOUT_MONO = (kFRONT_CENTER),
        kLAYOUT_STEREO = (kFRONT_LEFT | kFRONT_RIGHT),
        kLAYOUT_2POINT1 = (kLAYOUT_STEREO | kLOW_FREQUENCY),
        kLAYOUT_2_1 = (kLAYOUT_STEREO | kBACK_CENTER),
        kLAYOUT_SURROUND = (kLAYOUT_STEREO | kFRONT_CENTER),
        kLAYOUT_3POINT1 = (kLAYOUT_SURROUND | kLOW_FREQUENCY),
        kLAYOUT_4POINT0 = (kLAYOUT_SURROUND | kBACK_CENTER),
        kLAYOUT_4POINT1 = (kLAYOUT_4POINT0 | kLOW_FREQUENCY),
        kLAYOUT_2_2 = (kLAYOUT_STEREO | kSIDE_LEFT | kSIDE_RIGHT),
        kLAYOUT_QUAD = (kLAYOUT_STEREO | kBACK_LEFT | kBACK_RIGHT),
        kLAYOUT_5POINT0 = (kLAYOUT_SURROUND | kSIDE_LEFT | kSIDE_RIGHT),
        kLAYOUT_5POINT1 = (kLAYOUT_5POINT0 | kLOW_FREQUENCY),
        kLAYOUT_5POINT0_BACK = (kLAYOUT_SURROUND | kBACK_LEFT | kBACK_RIGHT),
        kLAYOUT_5POINT1_BACK = (kLAYOUT_5POINT0_BACK | kLOW_FREQUENCY),
        kLAYOUT_6POINT0 = (kLAYOUT_5POINT0 | kBACK_CENTER),
        kLAYOUT_6POINT0_FRONT =
            (kLAYOUT_2_2 | kFRONT_LEFT_OF_CENTER | kFRONT_RIGHT_OF_CENTER),
        kLAYOUT_HEXAGONAL = (kLAYOUT_5POINT0_BACK | kBACK_CENTER),
        kLAYOUT_6POINT1 = (kLAYOUT_5POINT1 | kBACK_CENTER),
        kLAYOUT_6POINT1_BACK = (kLAYOUT_5POINT1_BACK | kBACK_CENTER),
        kLAYOUT_6POINT1_FRONT = (kLAYOUT_6POINT0_FRONT | kLOW_FREQUENCY),
        kLAYOUT_7POINT0 = (kLAYOUT_5POINT0 | kBACK_LEFT | kBACK_RIGHT),
        kLAYOUT_7POINT0_FRONT =
            (kLAYOUT_5POINT0 | kFRONT_LEFT_OF_CENTER | kFRONT_RIGHT_OF_CENTER),
        kLAYOUT_7POINT1 = (kLAYOUT_5POINT1 | kBACK_LEFT | kBACK_RIGHT),
        kLAYOUT_7POINT1_WIDE =
            (kLAYOUT_5POINT1 | kFRONT_LEFT_OF_CENTER | kFRONT_RIGHT_OF_CENTER),
        kLAYOUT_7POINT1_WIDE_BACK =
            (kLAYOUT_5POINT1_BACK | kFRONT_LEFT_OF_CENTER |
             kFRONT_RIGHT_OF_CENTER),
        kLAYOUT_OCTAGONAL =
            (kLAYOUT_5POINT0 | kBACK_LEFT | kBACK_CENTER | kBACK_RIGHT),
        kLAYOUT_HEXADECAGONAL =
            (kLAYOUT_OCTAGONAL | kWIDE_LEFT | kWIDE_RIGHT | kTOP_BACK_LEFT |
             kTOP_BACK_RIGHT | kTOP_BACK_CENTER | kTOP_FRONT_CENTER |
             kTOP_FRONT_LEFT | kTOP_FRONT_RIGHT),
        kLAYOUT_STEREO_DOWNMIX = (kSTEREO_LEFT | kSTEREO_RIGHT)
    };
};

class BMF_API AudioFrame : public OpaqueDataSet, public SequenceData {
    struct Private;

    std::shared_ptr<Private> self;

  public:
    AudioFrame() = default;

    AudioFrame(const AudioFrame &) = default;

    AudioFrame(AudioFrame &&) = default;

    AudioFrame &operator=(const AudioFrame &) = default;

    AudioFrame(int samples, uint64_t layout, bool planer = true,
               const TensorOptions &options = kUInt8);

    AudioFrame(const TensorList &data, uint64_t layout, bool planer = true);

    static AudioFrame make(int samples, uint64_t layout, bool planer = true) {
        return AudioFrame(samples, layout, planer, TensorOptions(kUInt8));
    }

    template <typename... Options>
    static AudioFrame make(int samples, uint64_t layout, bool planer,
                           Options &&...opts) {
        return AudioFrame(
            samples, layout, planer,
            TensorOptions(kUInt8).options(std::forward<Options>(opts)...));
    }

    static AudioFrame make(const TensorList &data, uint64_t layout,
                           bool planer = true) {
        return AudioFrame(data, layout, planer);
    }

    AudioFrame clone() const;

    operator bool() const;

    uint64_t layout() const;

    ScalarType dtype() const;

    bool planer() const;

    int nsamples() const;

    int nchannels() const;

    void set_sample_rate(float sample_rate);

    float sample_rate() const;

    const TensorList &planes() const;

    int nplanes() const;

    Tensor plane(int p = 0) const;

    Tensor operator[](int p) const;

    /**
     * @brief copy all extra props(set by member func set_xxx) from
     * `from`(deepcopy if needed),
     *
     * @param from
     * @return AudioFrame&
     */
    AudioFrame &copy_props(const AudioFrame &from);
};

} // namespace bmf_sdk

BMF_DEFINE_TYPE(bmf_sdk::AudioFrame)