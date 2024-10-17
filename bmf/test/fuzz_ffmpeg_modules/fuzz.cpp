/*
 * Copyright 2024 Babit Authors
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
#include <filesystem>
#include <tuple>
#include <gtest/gtest.h>
#include <fuzztest/fuzztest.h>
#include <nlohmann/json.hpp>

#include <bmf/sdk/bmf.h>
#include <bmf/sdk/common.h>
#include <bmf/sdk/video_frame.h>
#include <bmf/sdk/json_param.h>
#include <bmf/sdk/module.h>
#include <bmf/sdk/module_functor.h>
#include <bmf/sdk/ffmpeg_helper.h>
#include <bmf/sdk/module_registry.h>
#include <bmf/sdk/packet.h>

USE_BMF_SDK_NS
namespace fs = std::filesystem; 

using namespace fuzztest;

namespace {
Domain<std::tuple<int, int, int, int>> AnyCrop(int width, int height) {
    auto valid_crop = [&](int x, int y) {
        return TupleOf(Just(x), Just(y), InRange(2, width-x), InRange(2, height-y));
    };
    return FlatMap(valid_crop, InRange(0, width-3), InRange(0, height-3));
}

auto AnyPreset() {
    return ElementOf<std::string>({
        "ultrafast",
        "superfast",
        "veryfast",
        "faster",
        "fast",
        "medium", 
        "slow",
        "slower",
        "veryslow",
        "placebo",
    });
}
}

void fuzz_decode_encode(std::tuple<int, int, int, int> crop, int crf, std::string preset) {
    BMFLOG_SET_LEVEL(BMF_INFO);

    int x, y, width, height; 
    std::tie(x, y, width, height) = crop;

    nlohmann::json decoder_para;
    decoder_para["input_path"] = "../../files/big_bunny_10s_30fps.mp4";
    auto decoder = make_sync_func<std::tuple<>, std::tuple<VideoFrame>>(
        ModuleInfo("CFFDecoder", "c++", ""), JsonParam(decoder_para)
    );

    nlohmann::json video_para, audio_para;
    video_para["codec"] = "h264";
    video_para["width"] = width;
    video_para["height"] = height;
    video_para["crf"] = crf;
    video_para["preset"] = preset;
    audio_para["codec"] = "aac";
    audio_para["bit_rate"] = 128000;
    audio_para["sample_rate"] = 44100;
    audio_para["channels"] = 2;
    nlohmann::json encoder_para;
    encoder_para["output_path"] = "./output.mp4";
    encoder_para["video_params"] = video_para;
    encoder_para["audio_params"] = audio_para;
    auto encoder = make_sync_func<std::tuple<VideoFrame>, std::tuple<>>(
        ModuleInfo("CFFEncoder", "c++", ""), JsonParam(encoder_para)
    );

    VideoFrame decoded_vf;
    while (decoded_vf.pts() != BMF_EOF) {
        std::tie(decoded_vf) = decoder();
        decoded_vf.crop(x, y, width, height);
        encoder(decoded_vf);
    }
}

FUZZ_TEST(ffmpeg_module, fuzz_decode_encode)
    .WithDomains(AnyCrop(1920, 1080), InRange(0, 51), AnyPreset());