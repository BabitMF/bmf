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
#include "builder.hpp"
#include "nlohmann/json.hpp"

#include "cpp_test_helper.h"

TEST(cpp_sync_mode, sync_videoframe) {
    std::string output_file = "./videoframe.jpg";
    BMF_CPP_FILE_REMOVE(output_file);

    bmf::builder::Graph graph = bmf::builder::Graph(bmf::builder::NormalMode);

    // create decoder
    nlohmann::json decoder_option = {{"input_path", "../../files/overlay.png"}};
    auto decoder =
        graph.Sync(std::vector<int>{}, std::vector<int>{0},
                   bmf_sdk::JsonParam(decoder_option), "c_ffmpeg_decoder");

    // create scale
    nlohmann::json scale_option = {{"name", "scale"}, {"para", "320:240"}};
    auto scale =
        graph.Sync(std::vector<int>{0}, std::vector<int>{0},
                   bmf_sdk::JsonParam(scale_option), "c_ffmpeg_filter");

    // create encoder
    nlohmann::json encoder_option = {{"output_path", output_file},
                                     {"format", "mjpeg"},
                                     {"video_params", {{"codec", "jpg"}}}};
    auto encoder =
        graph.Sync(std::vector<int>{0}, std::vector<int>{},
                   bmf_sdk::JsonParam(encoder_option), "c_ffmpeg_encoder");

    // call init if necessary, otherwise we skip this step
    graph.Init(decoder);
    graph.Init(scale);
    graph.Init(encoder);

    // decode
    auto decoded_frames = decoder.ProcessPkts();

    // scale
    bmf::builder::SyncPackets input_scale;
    input_scale.Insert(0, decoded_frames[0]);
    auto scaled_frames = scale.ProcessPkts(input_scale);

    // encode
    bmf::builder::SyncPackets input_encode;
    input_encode.Insert(0, scaled_frames[0]);
    encoder.ProcessPkts(input_encode);

    encoder.SendEOF();

    // call close if necessary, otherwise we skip this step
    graph.Close(decoder);
    graph.Close(scale);
    graph.Close(encoder);

    BMF_CPP_FILE_CHECK(output_file, "./"
                                    "videoframe.jpg|240|320|0.04|IMAGE2|950000|"
                                    "4750|mjpeg|{\"fps\": \"25.0\"}");
}

TEST(cpp_sync_mode, sync_audioframe) {
    std::string output_file = "./audioframe.mp4";
    BMF_CPP_FILE_REMOVE(output_file);

    bmf::builder::Graph graph = bmf::builder::Graph(bmf::builder::NormalMode);

    // create decoder
    nlohmann::json decoder_option = {
        {"input_path", "../../files/big_bunny_10s_30fps.mp4"}};
    auto decoder = graph.Sync(std::vector<int>{}, std::vector<int>{1},
                              decoder_option, "c_ffmpeg_decoder");

    // create volume
    nlohmann::json volume_option = {{"name", "volume"}, {"para", "volume=3"}};
    auto volume = graph.Sync(std::vector<int>{0}, std::vector<int>{0},
                             volume_option, "c_ffmpeg_filter");

    // create encoder
    nlohmann::json encoder_option = {{"output_path", output_file}};
    auto encoder = graph.Sync(std::vector<int>{0, 1}, std::vector<int>{},
                              encoder_option, "c_ffmpeg_encoder");

    // decode and get audio frame
    auto decoded_frames =
        decoder.ProcessPkts(std::map<int, std::vector<Packet>>{});

    // volume
    std::map<int, std::vector<Packet>> input_volume;
    input_volume.insert(std::make_pair(0, decoded_frames[1]));
    auto volume_frames = volume.ProcessPkts(input_volume);

    // encode
    std::map<int, std::vector<Packet>> input_encode;
    input_encode.insert(std::make_pair(1, input_volume[0]));
    encoder.ProcessPkts(input_encode);

    // send eof to encoder
    encoder.SendEOF();

    BMF_CPP_FILE_CHECK(
        output_file,
        "./audioframe.mp4|0|0|0.024|MOV,MP4,M4A,3GP,3G2,MJ2|271000|795||{}");
}

TEST(cpp_sync_mode, sync_video_by_pkts) {
    std::string output_file = "./video_simple_interface.mp4";
    BMF_CPP_FILE_REMOVE(output_file);

    bmf::builder::Graph graph = bmf::builder::Graph(bmf::builder::NormalMode);

    // create sync modules
    nlohmann::json decoder_option = {
        {"input_path", "../../files/big_bunny_10s_30fps.mp4"}};
    auto decoder = graph.Sync(std::vector<int>{}, std::vector<int>{0, 1},
                              decoder_option, "c_ffmpeg_decoder");

    nlohmann::json scale_option = {{"name", "scale"}, {"para", "320:250"}};
    auto scale =
        graph.Sync(std::vector<int>{0}, std::vector<int>{0},
                   bmf_sdk::JsonParam(scale_option), "c_ffmpeg_filter");

    nlohmann::json volume_option = {{"name", "volume"}, {"para", "volume=3"}};
    auto volume = graph.Sync(std::vector<int>{0}, std::vector<int>{0},
                             volume_option, "c_ffmpeg_filter");

    nlohmann::json encoder_option = {{"output_path", output_file}};
    auto encoder = graph.Sync(std::vector<int>{0, 1}, std::vector<int>{},
                              encoder_option, "c_ffmpeg_encoder");

    // call init if necessary, otherwise we skip this step
    graph.Init(decoder);
    graph.Init(scale);
    graph.Init(volume);
    graph.Init(encoder);

    // process video/audio by sync mode
    while (1) {
        auto decoded_frames = decoder.ProcessPkts();
        bool has_next = false;
        for (const auto &stream : decoded_frames.packets) {
            if (!stream.second.empty()) {
                has_next = true;
                if (stream.first == 0) {
                    bmf::builder::SyncPackets input_scale;
                    input_scale.Insert(0, decoded_frames[0]);
                    auto scaled_frames = scale.ProcessPkts(input_scale);

                    bmf::builder::SyncPackets input_encoder;
                    input_encoder.Insert(0, scaled_frames[0]);
                    encoder.ProcessPkts(input_encoder);
                } else if (stream.first == 1) {
                    bmf::builder::SyncPackets input_volume;
                    input_volume.Insert(0, decoded_frames[1]);
                    auto volume_frames = volume.ProcessPkts(input_volume);

                    bmf::builder::SyncPackets input_encoder;
                    input_encoder.Insert(1, volume_frames[0]);
                    encoder.ProcessPkts(input_encoder);
                }
            }
        }
        if (!has_next) {
            break;
        }
    }

    // call close if necessary, otherwise we skip this step
    graph.Close(decoder);
    graph.Close(scale);
    graph.Close(volume);
    graph.Close(encoder);

    BMF_CPP_FILE_CHECK(output_file, "./"
                                    "video_simple_interface.mp4|250|320|10.008|"
                                    "MOV,MP4,M4A,3GP,3G2,MJ2|224724|281130|"
                                    "h264|{\"fps\": \"30\"}");
}

TEST(cpp_sync_mode, sync_audio) {
    std::string output_file = "./audio.mp4";
    BMF_CPP_FILE_REMOVE(output_file);

    bmf::builder::Graph graph = bmf::builder::Graph(bmf::builder::NormalMode);

    // create sync modules
    nlohmann::json decoder_option = {
        {"input_path", "../../files/big_bunny_10s_30fps.mp4"}};
    auto decoder = graph.Sync(std::vector<int>{}, std::vector<int>{1},
                              decoder_option, "c_ffmpeg_decoder");

    nlohmann::json volume_option = {{"name", "volume"}, {"para", "volume=3"}};
    auto volume = graph.Sync(std::vector<int>{0}, std::vector<int>{0},
                             volume_option, "c_ffmpeg_filter");

    nlohmann::json encoder_option = {{"output_path", output_file}};
    auto encoder = graph.Sync(std::vector<int>{0, 1}, std::vector<int>{},
                              encoder_option, "c_ffmpeg_encoder");

    // process video/audio by sync mode
    while (1) {
        auto decoded_frames = decoder.ProcessPkts();
        if (decoded_frames[1].empty()) {
            encoder.SendEOF();
            break;
        }

        bmf::builder::SyncPackets input_volume;
        input_volume.Insert(0, decoded_frames[1]);
        auto volume_frames = volume.ProcessPkts(input_volume);

        bmf::builder::SyncPackets input_encoder;
        input_encoder.Insert(1, volume_frames[0]);
        encoder.ProcessPkts(input_encoder);
    }

    BMF_CPP_FILE_CHECK(
        output_file,
        "./audio.mp4|0|0|10.008|MOV,MP4,M4A,3GP,3G2,MJ2|131882|166192||{}");
}

TEST(cpp_sync_mode, sync_video) {
    std::string output_file = "./video_simple_interface.mp4";
    BMF_CPP_FILE_REMOVE(output_file);

    bmf::builder::Graph graph = bmf::builder::Graph(bmf::builder::NormalMode);

    // create sync modules
    nlohmann::json decoder_option = {
        {"input_path", "../../files/big_bunny_10s_30fps.mp4"}};
    auto decoder = graph.Sync(std::vector<int>{}, std::vector<int>{0, 1},
                              decoder_option, "c_ffmpeg_decoder");

    nlohmann::json scale_option = {{"name", "scale"}, {"para", "320:250"}};
    auto scale =
        graph.Sync(std::vector<int>{0}, std::vector<int>{0},
                   bmf_sdk::JsonParam(scale_option), "c_ffmpeg_filter");

    nlohmann::json volume_option = {{"name", "volume"}, {"para", "volume=3"}};
    auto volume = graph.Sync(std::vector<int>{0}, std::vector<int>{0},
                             volume_option, "c_ffmpeg_filter");

    nlohmann::json encoder_option = {{"output_path", output_file}};
    auto encoder = graph.Sync(std::vector<int>{0, 1}, std::vector<int>{},
                              encoder_option, "c_ffmpeg_encoder");

    // call init if necessary, otherwise we skip this step
    graph.Init(decoder);
    graph.Init(scale);
    graph.Init(volume);
    graph.Init(encoder);

    // process video/audio by sync mode
    while (1) {
        auto decoded_frames =
            graph.Process(decoder, bmf::builder::SyncPackets());
        bool has_next = false;
        for (const auto &stream : decoded_frames.packets) {
            if (!stream.second.empty()) {
                has_next = true;
                if (stream.first == 0) {
                    bmf::builder::SyncPackets input_scale;
                    input_scale.Insert(0, decoded_frames[0]);
                    auto scaled_frames = graph.Process(scale, input_scale);

                    bmf::builder::SyncPackets input_encoder;
                    input_encoder.Insert(0, scaled_frames[0]);
                    graph.Process(encoder, input_encoder);
                    // encoder.ProcessPkts(input_encoder);
                } else if (stream.first == 1) {
                    bmf::builder::SyncPackets input_volume;
                    input_volume.Insert(0, decoded_frames[1]);
                    auto volume_frames = graph.Process(volume, input_volume);
                    // auto volume_frames = volume.ProcessPkts(input_volume);

                    bmf::builder::SyncPackets input_encoder;
                    input_encoder.Insert(1, volume_frames[0]);
                    graph.Process(encoder, input_encoder);
                    // encoder.ProcessPkts(input_encoder);
                }
            }
        }
        if (!has_next) {
            break;
        }
    }

    // call close if necessary, otherwise we skip this step
    graph.Close(decoder);
    graph.Close(scale);
    graph.Close(volume);
    graph.Close(encoder);

    BMF_CPP_FILE_CHECK(output_file, "./"
                                    "video_simple_interface.mp4|250|320|10.008|"
                                    "MOV,MP4,M4A,3GP,3G2,MJ2|224724|281130|"
                                    "h264|{\"fps\": \"30\"}");
}

TEST(cpp_sync_mode, sync_eof_flush_data) {
    bmf::builder::Graph graph = bmf::builder::Graph(bmf::builder::NormalMode);

    // create decoder
    nlohmann::json decoder_option = {
        {"input_path", "../../files/big_bunny_10s_30fps.mp4"}};
    auto decoder =
        graph.Sync(std::vector<int>{}, std::vector<int>{0},
                   bmf_sdk::JsonParam(decoder_option), "c_ffmpeg_decoder");

    auto decoded_frames = graph.Process(decoder, bmf::builder::SyncPackets());
    std::cout << "get vframe number:" << decoded_frames.packets.size()
              << std::endl;
    decoder.SendEOF();
    std::cout << "get vframe number after send eof:"
              << decoded_frames.packets.size() << std::endl;
    graph.Close(decoder);
}
