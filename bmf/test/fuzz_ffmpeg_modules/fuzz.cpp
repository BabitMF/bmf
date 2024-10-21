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
static VideoFrame decode_one_frame(const std::string &path) {
    JsonParam option;
    option.parse(fmt::format("{{\"input_path\": \"{}\"}}", path));
    auto decoder = make_sync_func<std::tuple<>, std::tuple<VideoFrame>>(
        ModuleInfo("c_ffmpeg_decoder"), option);

    VideoFrame vf;
    std::tie(vf) = decoder();
    return vf;
}

Domain<std::tuple<int, int, int, int>> AnyCrop() { // min size 4x4 in 1080p image
    static const int width = 1920, height = 1080; 
    auto valid_crop = [&](int x, int y) {
        return TupleOf(Just(x), Just(y), InRange(4, width-x), InRange(4, height-y));
    };
    return FlatMap(valid_crop, InRange(0, width-5), InRange(0, height-5));
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

    // decode from input file
    VideoFrame decoded_vf;
    EXPECT_NO_THROW(decoded_vf = decode_one_frame("../../files/big_bunny_10s_30fps.mp4"));
    ASSERT_TRUE(decoded_vf);

    // crop params
    int x, y, width, height; 
    std::tie(x, y, width, height) = crop;

    // crop decoded frame
    EXPECT_NO_THROW(decoded_vf.crop(x, y, width, height));

    // setup encoder
    nlohmann::json video_para, audio_para;
    video_para["codec"] = "h264";
    video_para["width"] = width;
    video_para["height"] = height;
    video_para["crf"] = crf;
    video_para["preset"] = preset;
    nlohmann::json encoder_para;
    encoder_para["output_path"] = "./output.mp4";
    encoder_para["video_params"] = video_para;
    auto encode_one_frame = make_sync_func<std::tuple<VideoFrame>, std::tuple<>>(
        ModuleInfo("c_ffmpeg_encoder"), JsonParam(encoder_para)
    );

    // encode frame
    if (width%2==0 && height%2==0) {
        EXPECT_NO_THROW(encode_one_frame(decoded_vf));
    } else {
        EXPECT_THROW(encode_one_frame(decoded_vf), std::exception);
    }
}

FUZZ_TEST(ffmpeg_module, fuzz_decode_encode)
    .WithDomains(AnyCrop(), InRange(0, 51), AnyPreset());