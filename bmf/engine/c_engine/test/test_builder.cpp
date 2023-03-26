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
#include <fstream>
#include "gtest/gtest.h"
#include <bmf/sdk/log.h>
#include "../include/common.h"
#include "../../connector/include/builder.hpp"

TEST(builder, decode_encode) {
    BMFLOG_SET_LEVEL(BMF_INFO);

    bmf_nlohmann::json decoder_para;
    decoder_para["input_path"] = "../files/img.mp4";

    bmf_nlohmann::json video_para, audio_para;
    video_para["codec"] = "h264";
    video_para["width"] = 320;
    video_para["height"] = 240;
    video_para["crf"] = 23;
    video_para["preset"] = "veryfast";
    audio_para["codec"] = "aac";
    audio_para["bit_rate"] = 128000;
    audio_para["sample_rate"] = 44100;
    audio_para["channels"] = 2;
    bmf_nlohmann::json encoder_para;
    encoder_para["output_path"] = "./output.mp4";
    encoder_para["video_params"] = video_para;
    encoder_para["audio_params"] = audio_para;

    auto graph = bmf::builder::Graph(bmf::builder::NormalMode, bmf_sdk::JsonParam());
    auto video = graph.Decode(bmf_sdk::JsonParam(decoder_para));

    graph.Encode(video["video"], video["audio"], bmf_sdk::JsonParam(encoder_para));

    std::cout << graph.Dump() << std::endl;
    graph.Run();
}

TEST(builder, decode_passthrough_encode) {
    BMFLOG_SET_LEVEL(BMF_INFO);

    bmf_nlohmann::json decoder_para;
    decoder_para["input_path"] = "../files/img.mp4";

    bmf_nlohmann::json video_para, audio_para;
    video_para["codec"] = "h264";
    video_para["width"] = 320;
    video_para["height"] = 240;
    video_para["crf"] = 23;
    video_para["preset"] = "veryfast";
    audio_para["codec"] = "aac";
    audio_para["bit_rate"] = 128000;
    audio_para["sample_rate"] = 44100;
    audio_para["channels"] = 2;
    bmf_nlohmann::json encoder_para;
    encoder_para["output_path"] = "./output.mp4";
    encoder_para["video_params"] = video_para;
    encoder_para["audio_params"] = audio_para;

    auto graph = bmf::builder::Graph(bmf::builder::NormalMode, bmf_sdk::JsonParam());
    auto video = graph.Decode(bmf_sdk::JsonParam(decoder_para));

    graph.Encode(video["video"].CppModule({}, "pass_through", bmf_sdk::JsonParam()), video["audio"],
                 bmf_sdk::JsonParam(encoder_para));

    std::cout << graph.Dump() << std::endl;
    graph.Run();
}

TEST(builder, decode_filter_encode) {
    BMFLOG_SET_LEVEL(BMF_INFO);
    bmf_nlohmann::json decoder_para;
    decoder_para["input_path"] = "../files/img.mp4";

    bmf_nlohmann::json video_para, audio_para;
    video_para["codec"] = "h264";
    video_para["width"] = 320;
    video_para["height"] = 240;
    video_para["crf"] = 23;
    video_para["preset"] = "veryfast";
    audio_para["codec"] = "aac";
    audio_para["bit_rate"] = 128000;
    audio_para["sample_rate"] = 44100;
    audio_para["channels"] = 2;
    bmf_nlohmann::json encoder_para;
    encoder_para["output_path"] = "./with_null_audio.mp4";
    encoder_para["video_params"] = video_para;
    encoder_para["audio_params"] = audio_para;

    auto graph = bmf::builder::Graph(bmf::builder::NormalMode, bmf_sdk::JsonParam());
    auto video = graph.Decode(bmf_sdk::JsonParam(decoder_para));

    auto audio = video["audio"].Atrim("start=0:end=6");

    graph.Encode(video["video"], audio, bmf_sdk::JsonParam(encoder_para));

    std::cout << graph.Dump() << std::endl;
    graph.Run();

}

#if 0  //FIXME: need go module sdk support
TEST(builder, custom_module) {
    google::InitGoogleLogging("main");
    google::SetStderrLogging(google::INFO);
    bmf_nlohmann::json decoder_para;
    decoder_para["input_path"] = "../files/img.mp4";

    bmf_nlohmann::json video_para, audio_para;
    video_para["codec"] = "h264";
    video_para["width"] = 320;
    video_para["height"] = 240;
    video_para["crf"] = 23;
    video_para["preset"] = "veryfast";
    audio_para["codec"] = "aac";
    audio_para["bit_rate"] = 128000;
    audio_para["sample_rate"] = 44100;
    audio_para["channels"] = 2;
    bmf_nlohmann::json encoder_para;
    encoder_para["output_path"] = "./output.mp4";
    encoder_para["video_params"] = video_para;
    encoder_para["audio_params"] = audio_para;

    auto graph = bmf::builder::Graph(bmf::builder::NormalMode, bmf_sdk::JsonParam());
    auto video = graph.Decode(bmf_sdk::JsonParam(decoder_para));

    graph.Encode(video["video"].GoModule({}, "GoPassThrough", bmf_sdk::JsonParam()), video["audio"],
                 bmf_sdk::JsonParam(encoder_para));

    std::cout << graph.Dump() << std::endl;
    graph.Run();

    google::ShutdownGoogleLogging();
}

#endif