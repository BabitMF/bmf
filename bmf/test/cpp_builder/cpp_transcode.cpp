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

#include "connector.hpp"
#include "graph_config.h"
#include <bmf/sdk/video_frame.h>
#include <bmf/sdk/bmf_av_packet.h>
#include <fstream>
#include "cpp_test_helper.h"

TEST(cpp_transcode, transcode_simple) {
    std::string output_file = "./simple.mp4";
    BMF_CPP_FILE_REMOVE(output_file);

    nlohmann::json graph_para = {{"dump_graph", 1}};
    auto graph = bmf::builder::Graph(bmf::builder::NormalMode,
                                     bmf_sdk::JsonParam(graph_para));

    nlohmann::json decode_para = {
        {"input_path", "../../files/big_bunny_10s_30fps.mp4"}};
    auto video = graph.Decode(bmf_sdk::JsonParam(decode_para));

    nlohmann::json encode_para = {{"output_path", output_file},
                                  {"video_params",
                                   {{"codec", "h264"},
                                    {"width", 320},
                                    {"height", 240},
                                    {"crf", 23},
                                    {"preset", "veryfast"}}},
                                  {"audio_params",
                                   {{"codec", "aac"},
                                    {"bit_rate", 128000},
                                    {"sample_rate", 44100},
                                    {"channels", 2}}}};

    graph.Encode(video["video"], video["audio"],
                 bmf_sdk::JsonParam(encode_para));

    graph.Run();

    BMF_CPP_FILE_CHECK(output_file, "../transcode/"
                                    "simple.mp4|240|320|10.008|MOV,MP4,M4A,3GP,"
                                    "3G2,MJ2|192235|240486|h264|{\"fps\": "
                                    "\"30.0662251656\"}");
}

TEST(cpp_transcode, transcode_video) {
    std::string output_file = "./video.mp4";
    BMF_CPP_FILE_REMOVE(output_file);

    nlohmann::json graph_para = {{"dump_graph", 1}};

    auto graph = bmf::builder::Graph(bmf::builder::NormalMode,
                                     bmf_sdk::JsonParam(graph_para));

    nlohmann::json decode_tail_para = {
        {"input_path", "../../files/header.mp4"}};
    auto tail = graph.Decode(bmf_sdk::JsonParam(decode_tail_para));

    nlohmann::json decode_header_para = {
        {"input_path", "../../files/header.mp4"}};
    auto header = graph.Decode(bmf_sdk::JsonParam(decode_header_para));

    nlohmann::json decode_main_para = {
        {"input_path", "../../files/big_bunny_10s_30fps.mp4"}};
    auto video = graph.Decode(bmf_sdk::JsonParam(decode_main_para));

    nlohmann::json decode_logo1_para = {
        {"input_path", "../../files/xigua_prefix_logo_x.mov"}};
    auto logo_1 =
        graph.Decode(bmf_sdk::JsonParam(decode_logo1_para))["video"].Scale(
            "320:144");

    nlohmann::json decode_logo2_para = {
        {"input_path", "../../files/xigua_loop_logo2_x.mov"}};
    auto logo_2 = graph.Decode(bmf_sdk::JsonParam(decode_logo2_para))["video"]
                      .Scale("320:144")
                      .Loop("loop=-1:size=991")
                      .Setpts("PTS+3.900/TB");

    auto main_video =
        video["video"]
            .Scale("1280:720")
            .Overlay({logo_1}, "repeatlast=0")
            .Overlay({logo_2}, "shortest=1:x=if(gte(t,3.900),960,NAN):y=0");

    auto concat_video =
        graph.Concat({header["video"].Scale("1280:720"), main_video,
                      tail["video"].Scale("1280:720")},
                     "n=3");

    auto concat_audio = graph.Concat(
        {header["audio"], video["audio"], tail["audio"]}, "a=1:n=3:v=0");

    nlohmann::json encode_para = {
        {"output_path", output_file},
        {"video_params",
         {{"codec", "h264"},
          {"width", 1280},
          {"height", 720},
          {"crf", 23},
          {"preset", "veryfast"},
          {"x264-params", "ssim=1:psnr=1"},
          {"vsync", "vfr"},
          {"max_fr", 60}}},
        {"audio_params",
         {{"codec", "aac"},
          {"bit_rate", 128000},
          {"sample_rate", 44100},
          {"channels", 2}}},
        {"mux_params",
         {{"fflags", "+igndts"},
          {"movflags", "+faststart+use_metadata_tags"},
          {"max_interleave_delta", "0"}}}};

    graph.Encode(concat_video, concat_audio, bmf_sdk::JsonParam(encode_para));

    graph.Run();

    BMF_CPP_FILE_CHECK(output_file, "../transcode/"
                                    "video.mp4|720|1280|16.045|MOV,MP4,M4A,3GP,"
                                    "3G2,MJ2|2504766|5023622|h264|{\"fps\": "
                                    "\"43.29\"}");
}

TEST(cpp_transcode, transcode_image) {
    std::string output_file = "./image.jpg";
    BMF_CPP_FILE_REMOVE(output_file);

    nlohmann::json graph_para = {{"dump_graph", 1}};

    auto graph = bmf::builder::Graph(bmf::builder::NormalMode,
                                     bmf_sdk::JsonParam(graph_para));

    nlohmann::json decode_para = {{"input_path", "../../files/overlay.png"}};
    auto video = graph.Decode(bmf_sdk::JsonParam(decode_para));

    nlohmann::json encode_para = {
        {"output_path", output_file},
        {"format", "mjpeg"},
        {"video_params", {{"codec", "jpg"}, {"width", 320}, {"height", 240}}}};

    graph.Encode(video["video"].Scale("320:240"),
                 bmf_sdk::JsonParam(encode_para));

    graph.Run();

    BMF_CPP_FILE_CHECK(output_file, "image.jpg|240|320|0.040000|IMAGE2|975400|"
                                    "4877|mjpeg|{\"fps\": \"25.0\"}");
}

TEST(cpp_transcode, transcode_option) {
    std::string output_file = "./option.mp4";
    BMF_CPP_FILE_REMOVE(output_file);

    nlohmann::json graph_para = {{"dump_graph", 1}};

    auto graph = bmf::builder::Graph(bmf::builder::NormalMode,
                                     bmf_sdk::JsonParam(graph_para));

    nlohmann::json decode_para = {
        {"input_path", "../../files/big_bunny_10s_30fps.mp4"},
        {"start_time", 2}};
    auto video = graph.Decode(bmf_sdk::JsonParam(decode_para));

    nlohmann::json encode_para = {
        {"output_path", output_file},
        {"video_params",
         {{"codec", "h264"},
          {"width", 1280},
          {"height", 720},
          {"crf", 23},
          {"preset", "fast"},
          {"x264-params", "ssim=1:psnr=1"}}},
        {"audio_params",
         {{"codec", "aac"},
          {"bit_rate", 128000},
          {"sample_rate", 44100},
          {"channels", 2}}},
        {"mux_params",
         {{"fflags", "+igndts"},
          {"movflags", "+faststart+use_metadata_tags"},
          {"max_interleave_delta", "0"}}}};

    video["video"].EncodeAsVideo(video["audio"],
                                 bmf_sdk::JsonParam(encode_para));

    graph.Run();

    BMF_CPP_FILE_CHECK(output_file, "../transcode/"
                                    "option.mp4|720|1280|8.008|MOV,MP4,M4A,3GP,"
                                    "3G2,MJ2|1039220|1040260|h264|{\"fps\": "
                                    "\"30.1796407186\"}");
}

TEST(cpp_transcode, transcode_audio) {
    std::string output_file = "./audio.mp4";
    BMF_CPP_FILE_REMOVE(output_file);

    nlohmann::json graph_para = {{"dump_graph", 1}};
    auto graph = bmf::builder::Graph(bmf::builder::NormalMode,
                                     bmf_sdk::JsonParam(graph_para));

    nlohmann::json decode_para = {
        {"input_path", "../../files/big_bunny_10s_30fps.mp4"},
    };
    auto video = graph.Decode(bmf_sdk::JsonParam(decode_para));

    nlohmann::json encode_para = {{"output_path", output_file},
                                  {"audio_params",
                                   {{"codec", "aac"},
                                    {"bit_rate", 128000},
                                    {"sample_rate", 44100},
                                    {"channels", 2}}}};

    graph.Encode(graph.NewPlaceholderStream(), video["audio"],
                 bmf_sdk::JsonParam(encode_para));

    graph.Run();

    BMF_CPP_FILE_CHECK(output_file, "../transcode/"
                                    "audio.mp4|0|0|10.008|MOV,MP4,M4A,3GP,3G2,"
                                    "MJ2|136092|166183||{}");
}

TEST(cpp_transcode, transcode_with_input_only_audio) {
    std::string output_file = "./output.mp4";
    BMF_CPP_FILE_REMOVE(output_file);

    nlohmann::json graph_para = {{"dump_graph", 1}};
    auto graph = bmf::builder::Graph(bmf::builder::NormalMode,
                                     bmf_sdk::JsonParam(graph_para));

    nlohmann::json decode_para = {
        {"input_path", "../../files/big_bunny_10s_only_audio.mp4"},
    };
    auto video = graph.Decode(bmf_sdk::JsonParam(decode_para));

    nlohmann::json encode_para = {{"output_path", output_file}};

    graph.Encode(graph.NewPlaceholderStream(), video["audio"],
                 bmf_sdk::JsonParam(encode_para));

    graph.Run();
    BMF_CPP_FILE_CHECK(output_file,
                       "|0|0|10.008|MOV,MP4,M4A,3GP,3G2,MJ2|132840|166183||{}");
}

TEST(cpp_transcode,
     transcode_with_encode_with_audio_stream_but_no_audio_frame) {
    std::string output_file = "./output.mp4";
    BMF_CPP_FILE_REMOVE(output_file);

    nlohmann::json graph_para = {{"dump_graph", 1}};
    auto graph = bmf::builder::Graph(bmf::builder::NormalMode,
                                     bmf_sdk::JsonParam(graph_para));

    nlohmann::json decode_para = {
        {"input_path", "../../files/big_bunny_10s_30fps_only_video.mp4"},
    };
    auto video = graph.Decode(bmf_sdk::JsonParam(decode_para));

    nlohmann::json encode_para = {{"output_path", output_file}};

    graph.Encode(video["video"], video["audio"],
                 bmf_sdk::JsonParam(encode_para));

    graph.Run();
}

TEST(cpp_transcode, transcode_with_null_audio) {
    std::string output_file = "./with_null_audio.mp4";
    BMF_CPP_FILE_REMOVE(output_file);

    nlohmann::json graph_para = {{"dump_graph", 1}};
    auto graph = bmf::builder::Graph(bmf::builder::NormalMode,
                                     bmf_sdk::JsonParam(graph_para));

    nlohmann::json decode_para = {
        {"input_path", "../../files/big_bunny_10s_30fps.mp4"},
    };
    auto video = graph.Decode(bmf_sdk::JsonParam(decode_para));
    auto audio_stream = graph.NewPlaceholderStream();
    nlohmann::json encode_para = {{"output_path", output_file},
                                  {"video_params",
                                   {{"codec", "h264"},
                                    {"width", 320},
                                    {"height", 240},
                                    {"crf", 23},
                                    {"preset", "veryfast"}}},
                                  {"audio_params",
                                   {{"codec", "aac"},
                                    {"bit_rate", 128000},
                                    {"sample_rate", 44100},
                                    {"channels", 2}}}};

    graph.Encode(video["video"], audio_stream, bmf_sdk::JsonParam(encode_para));

    graph.Run();
    BMF_CPP_FILE_CHECK(output_file, "../transcode/"
                                    "with_null_audio.mp4|240|320|10.0|MOV,MP4,"
                                    "M4A,3GP,3G2,MJ2|60438|75548|h264|{\"fps\":"
                                    " \"30.0662251656\"}");
}

// TEST(cpp_transcode, transcode_cb) {
//     std::string output_file = "./cb.mp4";
//     BMF_CPP_FILE_REMOVE(output_file);

//     nlohmann::json graph_para = {
//         {"dump_graph", 0}
//     };
//     auto graph = bmf::builder::Graph(bmf::builder::NormalMode,
//     bmf_sdk::JsonParam(graph_para));

//     nlohmann::json decode_para = {
//         {"input_path", "../../files/big_bunny_10s_30fps.mp4"},
//     };
//     auto video = graph.Decode(bmf_sdk::JsonParam(decode_para));
//     nlohmann::json encode_para = {
//         {"output_path", output_file},
//         {"video_params", {
//             {"codec", "h264"},
//             {"width", 320},
//             {"height", 240},
//             {"crf", 23},
//             {"preset", "veryfast"}
//         }},
//         {"audio_params", {
//             {"codec", "aac"},
//             {"bit_rate", 128000},
//             {"sample_rate", 44100},
//             {"channels", 2}
//         }}
//     };
//     auto cb = [](void *, BMFCBytes) -> BMFCBytes {
//         BMFLOG(BMF_INFO) << "test cb cpp";
//         uint8_t bytes[] = {97, 98, 99, 100, 101, 0};
//         return BMFCBytes{bytes, 6};
//     };
//     BMFCallbackInstance cb_ = nullptr;
//     create_callback(cb, nullptr, &cb_);
//     graph.Encode(video["video"], video["audio"],
//     bmf_sdk::JsonParam(encode_para)).AddCallback(0, *cb_);
//     //std::cout << testing::internal::GetCapturedStdout();
//     graph.Run();
//     BMF_CPP_FILE_CHECK(
//         output_file,
//         "../transcode/cb.mp4|240|320|7.615000|MOV,MP4,M4A,3GP,3G2,MJ2|366635|348991|h264|{\"fps\":
//         \"30.0662251656\"}"
//     );
// }

TEST(cpp_transcode, transcode_hls) {
    std::string output_file = "./file000.ts";
    BMF_CPP_FILE_REMOVE(output_file);

    nlohmann::json graph_para = {{"dump_graph", 1}};
    auto graph = bmf::builder::Graph(bmf::builder::NormalMode,
                                     bmf_sdk::JsonParam(graph_para));

    nlohmann::json decode_para = {
        {"input_path", "../../files/big_bunny_10s_30fps.mp4"},
    };
    auto video = graph.Decode(bmf_sdk::JsonParam(decode_para));
    nlohmann::json encode_para = {
        {"output_path", "./out.hls"},
        {"format", "hls"},
        {"mux_params",
         {{"hls_list_size", "0"},
          {"hls_time", "10"},
          {"hls_segment_filename", "./file%03d.ts"}}}};

    graph.Encode(video["video"], video["audio"],
                 bmf_sdk::JsonParam(encode_para));

    graph.Run();
    BMF_CPP_FILE_CHECK(output_file, "../transcode/"
                                    "file000.ts|1080|1920|10.0304|MPEGTS|"
                                    "2029494|2544580|h264|{\"fps\": "
                                    "\"29.97\"}");
}

TEST(cpp_transcode, transcode_crypt) {
    std::string output_file = "./crypt.mp4";
    BMF_CPP_FILE_REMOVE(output_file);

    nlohmann::json graph_para = {{"dump_graph", 1}};
    auto graph = bmf::builder::Graph(bmf::builder::NormalMode,
                                     bmf_sdk::JsonParam(graph_para));

    nlohmann::json decode_para = {
        {"input_path", "../../files/encrypt.mp4"},
        {"decryption_key", "b23e92e4d603468c9ec7be7570b16229"}};
    auto video = graph.Decode(bmf_sdk::JsonParam(decode_para));
    nlohmann::json encode_para = {
        {"output_path", output_file},
        {"mux_params",
         {{"encryption_scheme", "cenc-aes-ctr"},
          {"encryption_key", "76a6c65c5ea762046bd749a2e632ccbb"},
          {"encryption_kid", "a7e61c373e219033c21091fa607bf3b8"}}}};

    graph.Encode(video["video"], video["audio"],
                 bmf_sdk::JsonParam(encode_para));

    graph.Run();
    BMF_CPP_FILE_CHECK(output_file, "../transcode/"
                                    "crypt.mp4|640|360|10.076000|MOV,MP4,M4A,"
                                    "3GP,3G2,MJ2|991807|1249182|h264|{\"fps\": "
                                    "\"20.0828500414\"}");
}

TEST(cpp_transcode, transcode_short_video_concat) {
    std::string output_file = "./simple.mp4";
    std::string input_file_1 = "../../files/big_bunny_10s_30fps.mp4";
    std::string input_file_2 = "../../files/single_frame.mp4";
    nlohmann::json graph_para = {{"dump_graph", 1}};
    auto graph = bmf::builder::Graph(bmf::builder::NormalMode,
                                     bmf_sdk::JsonParam(graph_para));

    nlohmann::json decode_para = {{"input_path", input_file_1}};
    nlohmann::json decode_para_2 = {{"input_path", input_file_2}};
    auto video = graph.Decode(bmf_sdk::JsonParam(decode_para))["video"];
    auto video2 = graph.Decode(bmf_sdk::JsonParam(decode_para_2))["video"];
    auto vout = video.FFMpegFilter({video2}, "concat", "", "");
    nlohmann::json encode_para = {
        {"output_path", output_file},
        {"video_params",
         {{"codec", "h264"},
          {"width", 320},
          {"height", 240},
          {"crf", 23},
          {"preset", "veryfast"}}},
    };

    graph.Encode(vout, bmf_sdk::JsonParam(encode_para));

    graph.Run();
}

TEST(cpp_transcode, transcode_map_param) {
    std::string input_file = "../../files/big_bunny_multi_stream.mp4";
    std::string output_file_1 = "./output_1.mp4";
    std::string output_file_2 = "./output_2.mp4";
    nlohmann::json graph_para = {{"dump_graph", 1}};
    BMF_CPP_FILE_REMOVE(output_file_1);
    BMF_CPP_FILE_REMOVE(output_file_2);
    auto graph = bmf::builder::Graph(bmf::builder::NormalMode,
                                     bmf_sdk::JsonParam(graph_para));

    nlohmann::json decode_para = {
        {"input_path", input_file},
        {"map_v", 0},
        {"map_a", 2},
    };
    nlohmann::json decode_para_2 = {
        {"input_path", input_file},
        {"map_v", 1},
        {"map_a", 3},
    };
    nlohmann::json encode_para_1 = {
        {"output_path", output_file_1},
    };
    nlohmann::json encode_para_2 = {
        {"output_path", output_file_2},
    };
    auto video1 = graph.Decode(bmf_sdk::JsonParam(decode_para));
    graph.Encode(video1["video"], video1["audio"],
                 bmf_sdk::JsonParam(encode_para_1));
    auto video2 = graph.Decode(bmf_sdk::JsonParam(decode_para_2));
    graph.Encode(video2["video"], video2["audio"],
                 bmf_sdk::JsonParam(encode_para_2));
    graph.Run();
    BMF_CPP_FILE_CHECK(output_file_1, "../transcode/"
                                      "output_1.mp4|720|1280|10.008|MOV,MP4,"
                                      "M4A,3GP,3G2,MJ2|828031|1035868|h264|{"
                                      "\"fps\": \"30.10\"}");
    BMF_CPP_FILE_CHECK(output_file_2, "../transcode/"
                                      "output_2.mp4|1080|1920|10.008|MOV,MP4,"
                                      "M4A,3GP,3G2,MJ2|1822167|2283859|h264|{"
                                      "\"fps\": \"30.10\"}");
}

TEST(cpp_transcode, transcode_rgb_2_video) {
    std::string input_file = "../../files/test_rgba_806x654.rgb";
    std::string output_file = "./rgb2video.mp4";
    nlohmann::json graph_para = {{"dump_graph", 1}};
    BMF_CPP_FILE_REMOVE(output_file);
    auto graph = bmf::builder::Graph(bmf::builder::NormalMode,
                                     bmf_sdk::JsonParam(graph_para));

    nlohmann::json decode_para = {
        {"input_path", input_file},
        {"s", "806:654"},
        {"pix_fmt", "rgba"},
    };
    nlohmann::json encode_para = {
        {"output_path", output_file},
    };
    auto video_stream =
        graph.Decode(bmf_sdk::JsonParam(decode_para))["video"].Loop(
            "loop=50:size=1");
    video_stream.EncodeAsVideo(bmf_sdk::JsonParam(encode_para));
    graph.Run();
    BMF_CPP_FILE_CHECK(output_file, "../transcode/"
                                    "rgb2video.mp4|654|806|2.04|MOV,MP4,M4A,"
                                    "3GP,3G2,MJ2|58848|15014|h264|{\"fps\": "
                                    "\"25.0\"}");
}

TEST(cpp_transcode, transcode_stream_copy) {
    std::string input_file = "../../files/big_bunny_10s_30fps_mpeg.mp4";
    std::string output_file = "./stream_copy.mp4";
    nlohmann::json graph_para = {{"dump_graph", 1}};
    BMF_CPP_FILE_REMOVE(output_file);
    auto graph = bmf::builder::Graph(bmf::builder::NormalMode,
                                     bmf_sdk::JsonParam(graph_para));

    nlohmann::json decode_para = {{"input_path", input_file},
                                  {"video_codec", "copy"}};
    nlohmann::json encode_para = {
        {"output_path", output_file},
    };
    auto video = graph.Decode(bmf_sdk::JsonParam(decode_para));
    auto video_stream = video["video"];
    video_stream.EncodeAsVideo(video["audio"], bmf_sdk::JsonParam(encode_para));
    graph.Run();
    BMF_CPP_FILE_CHECK(output_file, "../transcode/"
                                    "stream_copy.mp4|1080|1920|10.008|MOV,MP4,"
                                    "M4A,3GP,3G2,MJ2|2255869|2822093|mpeg4|{"
                                    "\"fps\": \"29.97\"}");
}

TEST(cpp_transcode, transcode_stream_audio_copy) {
    std::string input_file = "../../files/big_bunny_10s_only_audio.flv";
    std::string output_file = "./audio_copy.mp4";
    nlohmann::json graph_para = {{"dump_graph", 1}};
    BMF_CPP_FILE_REMOVE(output_file);
    auto graph = bmf::builder::Graph(bmf::builder::NormalMode,
                                     bmf_sdk::JsonParam(graph_para));

    nlohmann::json decode_para = {{"input_path", input_file},
                                  {"video_codec", "copy"},
                                  {"audio_codec", "copy"}};
    nlohmann::json encode_para = {
        {"output_path", output_file},
    };
    auto video = graph.Decode(bmf_sdk::JsonParam(decode_para));
    graph.Encode(graph.NewPlaceholderStream(), video["audio"],
                 bmf_sdk::JsonParam(encode_para));
    graph.Run();
    BMF_CPP_FILE_CHECK(output_file, "../transcode/"
                                    "audio_copy.mp4|0|0|10.031|MOV,MP4,M4A,3GP,"
                                    "3G2,MJ2|129999|163003|{\"accurate\": "
                                    "\"b\"}");
}

TEST(cpp_transcode, transcode_extract_frames) {
    std::string input_file = "../../files/big_bunny_10s_30fps.mp4";
    std::string output_file = "./audio_copy.mp4";
    nlohmann::json graph_para = {{"dump_graph", 1}};
    BMF_CPP_FILE_REMOVE(output_file);
    auto graph = bmf::builder::Graph(bmf::builder::GeneratorMode,
                                     bmf_sdk::JsonParam(graph_para));

    nlohmann::json decode_para = {
        {"input_path", input_file},
        {"video_params", {{"extract_frames", {{"fps", 0.5}}}}}
        //{"video_params", {"extract_frames", { "fps", 0.5}}}
    };
    nlohmann::json encode_para = {
        {"output_path", output_file},
    };
    auto videoStream = graph.Decode(bmf_sdk::JsonParam(decode_para))["video"];
    videoStream.Start();
    int num = 0;
    while (true) {
        Packet pkt = graph.Generate(videoStream.GetName());
        if (pkt.timestamp() == BMF_EOF) {
            break;
        }
        if (pkt.is<VideoFrame>() == true) {
            VideoFrame video_frame = pkt.get<VideoFrame>();
            num++;
        }
    }
    EXPECT_EQ(num, 6);
}

TEST(cpp_transcode, transcode_incorrect_stream_notify) {
    std::string input_file = "../../files/big_bunny_10s_30fps.mp4";
    std::string output_file = "./incorrect_stream_notify.mp4";
    nlohmann::json graph_para = {{"dump_graph", 1}};
    BMF_CPP_FILE_REMOVE(output_file);
    auto graph = bmf::builder::Graph(bmf::builder::NormalMode,
                                     bmf_sdk::JsonParam(graph_para));

    nlohmann::json decode_para = {{"input_path", input_file}};
    nlohmann::json encode_para = {
        {"output_path", output_file},
    };
    auto video = graph.Decode(bmf_sdk::JsonParam(decode_para));
    try {
        auto v = video["wrong_name"];
        graph.Encode(v, bmf_sdk::JsonParam(encode_para));
        graph.Run();
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
    }
}

TEST(cpp_transcode, transcode_incorrect_encoder_param) {
    std::string input_file = "../../files/big_bunny_10s_30fps.mp4";
    std::string output_file = "./incorrect_encoder_param.mp4";
    nlohmann::json graph_para = {{"dump_graph", 1}};
    BMF_CPP_FILE_REMOVE(output_file);
    auto graph = bmf::builder::Graph(bmf::builder::NormalMode,
                                     bmf_sdk::JsonParam(graph_para));

    nlohmann::json decode_para = {{"input_path", input_file}};
    auto video = graph.Decode(bmf_sdk::JsonParam(decode_para));
    auto v = video["video"];
    auto a = video["audio"];
    std::string wrong_k_1 = "wrong_key_1";
    std::string wrong_v_1 = "wrong_value_1";
    std::string wrong_k_2 = "wrong_key_2";
    std::string wrong_v_2 = "wrong_value_2";
    nlohmann::json encode_para = {
        {"output_path", output_file},
        {"video_params",
         {
             {"codec", "h264"},
             {"preset", "fast"},
             {"crf", "23"},
             {wrong_k_1, wrong_v_1},
             {wrong_k_2, wrong_v_2},
         }},
        {"audio_params", {{wrong_k_1, wrong_v_1}, {wrong_k_2, wrong_v_2}}},
        {"mux_params", {{wrong_k_1, wrong_v_1}, {wrong_k_2, wrong_v_2}}}};
    try {
        graph.Encode(v, a, bmf_sdk::JsonParam(encode_para));
        graph.Run();
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
    }
}

TEST(cpp_transcode, transcode_duration) {
    std::string input_file = "../../files/big_bunny_10s_30fps.mp4";
    std::string output_file = "./durations.mp4";
    nlohmann::json graph_para = {{"dump_graph", 1}};
    BMF_CPP_FILE_REMOVE(output_file);
    auto graph = bmf::builder::Graph(bmf::builder::NormalMode,
                                     bmf_sdk::JsonParam(graph_para));

    nlohmann::json decode_para = {{"input_path", input_file},
                                  {"durations", {1.5, 3, 5, 6}}};
    auto video = graph.Decode(bmf_sdk::JsonParam(decode_para));
    nlohmann::json encode_para = {
        {"output_path", output_file},
        {"video_params",
         {{"codec", "h264"},
          {"width", 320},
          {"height", 240},
          {"crf", 23},
          {"preset", "veryfast"},
          {"vsync", "vfr"},
          {"r", 30}}},
    };
    graph.Encode(video["video"], video["audio"],
                 bmf_sdk::JsonParam(encode_para));
    graph.Run();
    BMF_CPP_FILE_CHECK(output_file, "../transcode/"
                                    "durations.mp4|240|320|4.54|MOV,MP4,M4A,"
                                    "3GP,3G2,MJ2|105089|59113|h264|{\"fps\": "
                                    "\"16.67\"}");
}

TEST(cpp_transcode, transcode_output_raw_video) {
    std::string input_file = "../../files/big_bunny_10s_30fps.mp4";
    std::string output_file = "./out.yuv";
    nlohmann::json graph_para = {{"dump_graph", 1}};
    BMF_CPP_FILE_REMOVE(output_file);
    auto graph = bmf::builder::Graph(bmf::builder::NormalMode,
                                     bmf_sdk::JsonParam(graph_para));

    nlohmann::json decode_para = {{"input_path", input_file}};
    auto video = graph.Decode(bmf_sdk::JsonParam(decode_para));
    nlohmann::json encode_para = {
        {"output_path", output_file},
        {"video_params",
         {
             {"codec", "rawvideo"},
         }},
        {"format", "rawvideo"},
    };
    graph.Encode(video["video"], bmf_sdk::JsonParam(encode_para));
    graph.Run();
    BMF_CPP_FILE_CHECK_MD5(output_file, "992f929388f18c43c06c767d63eea15d");
}

TEST(cpp_transcode, transcode_output_null) {
    std::string input_file = "../../files/big_bunny_10s_30fps.mp4";
    nlohmann::json graph_para = {{"dump_graph", 1}};
    auto graph = bmf::builder::Graph(bmf::builder::NormalMode,
                                     bmf_sdk::JsonParam(graph_para));

    nlohmann::json decode_para = {{"input_path", input_file}};
    auto video = graph.Decode(bmf_sdk::JsonParam(decode_para));
    nlohmann::json encode_para = {{"null_output", 1}};
    graph.Encode(video["video"], video["audio"],
                 bmf_sdk::JsonParam(encode_para));
    graph.Run();
}

TEST(cpp_transcode, transcode_vframes) {
    std::string input_file = "../../files/big_bunny_10s_30fps.mp4";
    std::string output_file = "./simple.mp4";
    nlohmann::json graph_para = {{"dump_graph", 1}};
    BMF_CPP_FILE_REMOVE(output_file);
    auto graph = bmf::builder::Graph(bmf::builder::NormalMode,
                                     bmf_sdk::JsonParam(graph_para));

    nlohmann::json decode_para = {
        {"input_path", input_file},
    };
    auto video = graph.Decode(bmf_sdk::JsonParam(decode_para));
    nlohmann::json encode_para = {
        {"output_path", output_file},
        {"vframes", 30},
        {"video_params",
         {{"codec", "h264"},
          {"width", 640},
          {"height", 480},
          {"crf", 23},
          {"preset", "veryfast"}}},
    };
    graph.Encode(video["video"], bmf_sdk::JsonParam(encode_para));
    graph.Run();
    BMF_CPP_FILE_CHECK(output_file, "../transcode/"
                                    "simple.mp4|480|640|1.001000|MOV,MP4,M4A,"
                                    "3GP,3G2,MJ2|110976|13872|h264|{\"fps\": "
                                    "\"29.97\"}");
}

TEST(cpp_transcode, transcode_segment_trans) {
    std::string input_file = "../../files/big_bunny_10s_30fps_mpeg.mp4";
    std::string output_file = "./simple_%05d.mp4";
    nlohmann::json graph_para = {{"dump_graph", 1}};
    BMF_CPP_FILE_REMOVE(output_file);
    auto graph = bmf::builder::Graph(bmf::builder::NormalMode,
                                     bmf_sdk::JsonParam(graph_para));

    nlohmann::json decode_para = {{"input_path", input_file},
                                  {"video_codec", "copy"}};
    auto video = graph.Decode(bmf_sdk::JsonParam(decode_para));
    nlohmann::json encode_para = {
        {"output_path", output_file},
        {"format", "segment"},
        {"mux_params", {{"segment_time", 4}}},
    };
    graph.Encode(video["video"], video["audio"],
                 bmf_sdk::JsonParam(encode_para));
    graph.Run();
    BMF_CPP_FILE_CHECK("./simple_00000.mp4", "../"
                                             "simple_00000.mp4|1080|1920|4.296|"
                                             "MOV,MP4,M4A,3GP,3G2,MJ2|1988878|"
                                             "1068028|mpeg4|{\"fps\": "
                                             "\"29.97\", \"accurate\": \"d\"}");
    BMF_CPP_FILE_CHECK("./simple_00001.mp4", "../transcode/"
                                             "simple_00001.mp4|1080|1920|8.313|"
                                             "MOV,MP4,M4A,3GP,3G2,MJ2|1102862|"
                                             "1146012|mpeg4|{\"fps\": "
                                             "\"29.97\", \"accurate\": \"d\"}");
}

TEST(cpp_transcode, test_encoder_push_output_mp4) {
    std::string input_file = "../../files/big_bunny_10s_30fps.mp4";
    std::string output_file = "./simple_vframe_python.mp4";
    nlohmann::json graph_para = {{"dump_graph", 1}};
    BMF_CPP_FILE_REMOVE(output_file);
    auto graph = bmf::builder::Graph(bmf::builder::GeneratorMode,
                                     bmf_sdk::JsonParam(graph_para));

    nlohmann::json decode_para = {{"input_path", input_file}};
    auto video = graph.Decode(bmf_sdk::JsonParam(decode_para));
    nlohmann::json encode_para = {{"output_path", output_file},
                                  {"push_output", 1},
                                  {"vframes", 60},
                                  {"video_params",
                                   {{"codec", "jpg"},
                                    {"width", 640},
                                    {"height", 480},
                                    {"crf", 23},
                                    {"preset", "veryfast"}}}};
    nlohmann::json encode_para_2 = {{"output_path", output_file}};
    bmf::builder::Stream stream =
        graph.Encode(video["video"], bmf_sdk::JsonParam(encode_para));
    stream.Start();
    std::ofstream outFile(output_file, std::ios::out | std::ios::binary);
    while (true) {
        Packet pkt = graph.Generate(stream.GetName());
        if (pkt.timestamp() == BMF_EOF) {
            break;
        }
        if (pkt.is<BMFAVPacket>() == true) {
            BMFAVPacket ppkt = pkt.get<BMFAVPacket>();
            int offset = ppkt.get_offset();
            int whence = ppkt.get_whence();
            void *data = ppkt.data_ptr();
            if (offset > 0) {
                if (whence == 0) {
                    outFile.seekp(offset, std::ios_base::beg);
                } else if (whence == 1) {
                    outFile.seekp(offset, std::ios_base::cur);
                } else if (whence == 2) {
                    outFile.seekp(offset, std::ios_base::end);
                }
            }
            std::cout << "tby : " << offset << " "
                      << " " << whence << " " << ppkt.nbytes() << std::endl;
            outFile.write((char *)data, ppkt.nbytes());
        }
    }
    outFile.close();
}

TEST(cpp_transcode, test_encoder_push_output_image2pipe) {
    std::string input_file = "../../files/big_bunny_10s_30fps.mp4";
    nlohmann::json graph_para = {{"dump_graph", 1}};
    auto graph = bmf::builder::Graph(bmf::builder::GeneratorMode,
                                     bmf_sdk::JsonParam(graph_para));

    nlohmann::json decode_para = {{"input_path", input_file}};
    auto video = graph.Decode(bmf_sdk::JsonParam(decode_para));
    nlohmann::json encode_para = {{"push_output", 1},
                                  {"vframes", 2},
                                  {"format", "image2pipe"},
                                  {"avio_buffer_size", 65536},
                                  {"video_params",
                                   {{"codec", "jpg"},
                                    {"width", 640},
                                    {"height", 480},
                                    {"crf", 23},
                                    {"preset", "veryfast"}}}};

    bmf::builder::Stream stream =
        graph.Encode(video["video"], bmf_sdk::JsonParam(encode_para));
    stream.Start();
    int write_num = 0;
    while (true) {
        Packet pkt = graph.Generate(stream.GetName());
        if (pkt.timestamp() == BMF_EOF) {
            break;
        }
        if (pkt.is<BMFAVPacket>() == true) {
            BMFAVPacket ppkt = pkt.get<BMFAVPacket>();
            int offset = ppkt.get_offset();
            int whence = ppkt.get_whence();
            void *data = ppkt.data_ptr();
            std::string output_file =
                "./simple_image" + std::to_string(write_num) + ".jpg";
            std::ofstream outFile(output_file,
                                  std::ios::out | std::ios::binary);
            if (offset > 0) {
                if (whence == 0) {
                    outFile.seekp(offset, std::ios_base::beg);
                } else if (whence == 1) {
                    outFile.seekp(offset, std::ios_base::cur);
                } else if (whence == 2) {
                    outFile.seekp(offset, std::ios_base::end);
                }
            }
            outFile.write((char *)data, ppkt.nbytes());
            write_num++;
            outFile.close();
        }
    }
}

TEST(cpp_transcode, test_encoder_push_output_audio_pcm_s16le) {
    std::string input_file = "../../files/big_bunny_10s_30fps.mp4";
    std::string output_file = "./test_audio_simple_pcm_s16le.wav";
    nlohmann::json graph_para = {{"dump_graph", 1}};
    BMF_CPP_FILE_REMOVE(output_file);
    auto graph = bmf::builder::Graph(bmf::builder::GeneratorMode,
                                     bmf_sdk::JsonParam(graph_para));

    nlohmann::json decode_para = {{"input_path", input_file}};
    auto video = graph.Decode(bmf_sdk::JsonParam(decode_para));
    nlohmann::json encode_para = {{"output_path", output_file},
                                  {"push_output", 1},
                                  {"format", "wav"},
                                  {"audio_params",
                                   {
                                       {"codec", "pcm_s16le"},
                                   }}};
    nlohmann::json encode_para_2 = {{"output_path", output_file}};
    bmf::builder::Stream stream =
        graph.Encode(graph.NewPlaceholderStream(), video["audio"],
                     bmf_sdk::JsonParam(encode_para));
    stream.Start();
    std::ofstream outFile(output_file, std::ios::out | std::ios::binary);
    while (true) {
        Packet pkt = graph.Generate(stream.GetName());
        if (pkt.timestamp() == BMF_EOF) {
            break;
        }
        if (pkt.is<BMFAVPacket>() == true) {
            auto ppkt = pkt.get<BMFAVPacket>();
            int offset = ppkt.get_offset();
            int whence = ppkt.get_whence();
            void *data = ppkt.data_ptr();
            if (offset > 0) {
                if (whence == 0) {
                    outFile.seekp(offset, std::ios_base::beg);
                } else if (whence == 1) {
                    outFile.seekp(offset, std::ios_base::cur);
                } else if (whence == 2) {
                    outFile.seekp(offset, std::ios_base::end);
                }
            }
            outFile.write((char *)data, ppkt.nbytes());
        }
    }
    outFile.close();
}

TEST(cpp_transcode, test_generator) {
    std::string input_file = "../../files/big_bunny_10s_30fps.mp4";
    nlohmann::json graph_para = {{"dump_graph", 1}};
    auto graph = bmf::builder::Graph(bmf::builder::GeneratorMode,
                                     bmf_sdk::JsonParam(graph_para));

    nlohmann::json decode_para = {{"input_path", input_file}};

    auto stream =
        graph.Decode(bmf_sdk::JsonParam(decode_para))["video"].Scale("299:299");
    stream.Start();
    int frame_num = 0;
    while (true) {
        Packet pkt = graph.Generate(stream[0].GetName());
        if (pkt.timestamp() == BMF_EOF) {
            break;
        }
        if (pkt.is<VideoFrame>() == true) {
            VideoFrame video_frame = pkt.get<VideoFrame>();
            std::cout << "frame :" << frame_num << " "
                      << "format: " << video_frame.frame().format()
                      << std::endl;
        }
    }
}

/* TEST(cpp_transcode, transcode_encoder_push_unmuxed_output_mp4) { */
/*     std::string input_file = "../../files/big_bunny_10s_30fps.mp4"; */
/*     std::string output_file = "./unmuxed_output.mp4"; */
/*     nlohmann::json graph_para = { */
/*         {"dump_graph", 1} */
/*     }; */
/*     BMF_CPP_FILE_REMOVE(output_file); */
/*     auto graph = bmf::builder::Graph(bmf::builder::NormalMode,
 * bmf_sdk::JsonParam(graph_para)); */

/*     nlohmann::json decode_para = { */
/*         {"input_path", input_file} */
/*     }; */
/*     auto video = graph.Decode(bmf_sdk::JsonParam(decode_para)); */
/*     nlohmann::json encode_para = { */
/*         {"output_path", output_file}, */
/*         {"push_output", 2}, */
/*         {"video_params", { */
/*             {"codec", "h264"}, */
/*             {"width", 320}, */
/*             {"height", 240}, */
/*             {"crf", 23}, */
/*             {"preset", "veryfast"} */
/*         }}, */
/*         {"audio_params", { */
/*             {"codec", "aac"}, */
/*             {"bit_rate", 128000}, */
/*             {"sample_rate", 44100}, */
/*             {"channels", 2} */
/*         }} */
/*     }; */
/*     nlohmann::json encode_para_2 = { */
/*         {"output_path", output_file} */
/*     }; */
/*     auto encoded_video = graph.Encode(video["video"], video["audio"],
 * bmf_sdk::JsonParam(encode_para)); */
/*     graph.Encode(encoded_video["video"], encoded_video["audio"],
 * bmf_sdk::JsonParam(encode_para_2)); */
/*     graph.Run(); */
/*     BMF_CPP_FILE_CHECK( */
/*         output_file, */
/*         "../transcode/unmuxed_output.mp4|240|320|7.574233|MOV,MP4,M4A,3GP,3G2,MJ2|404941|384340|h264|{\"fps\":
 * \"29.97002997\"}" */
/*     ); */
/* } */

/* TEST(cpp_transcode, transcode_push_pkt_into_decoder) { */
/*     std::string input_file = "../../files/big_bunny_10s_30fps.mp4"; */
/*     std::string output_file = "./aac.mp4"; */
/*     nlohmann::json graph_para = { */
/*         {"dump_graph", 1} */
/*     }; */
/*     BMF_CPP_FILE_REMOVE(output_file); */
/*     auto graph = bmf::builder::Graph(bmf::builder::UpdateMode,
 * bmf_sdk::JsonParam(graph_para)); */

/*     nlohmann::json decode_para = { */
/*         {"input_path", input_file} */
/*     }; */
/*     auto video = graph.Decode(bmf_sdk::JsonParam(decode_para)); */
/*     nlohmann::json encode_para = { */
/*         {"output_path", output_file}, */
/*         {"push_output", 2}, */
/*         {"video_params", { */
/*             {"codec", "h264"}, */
/*             {"width", 320}, */
/*             {"height", 240}, */
/*             {"crf", 23}, */
/*             {"preset", "veryfast"} */
/*         }}, */
/*         {"audio_params", { */
/*             {"codec", "aac"}, */
/*             {"bit_rate", 128000}, */
/*             {"sample_rate", 44100}, */
/*             {"channels", 2} */
/*         }} */
/*     }; */
/*     nlohmann::json encode_para_2 = { */
/*         {"output_path", output_file} */
/*     }; */
/*     auto encoded_video = graph.Encode(video["video"], video["audio"],
 * bmf_sdk::JsonParam(encode_para)); */
/*     graph.Encode(encoded_video["video"], encoded_video["audio"],
 * bmf_sdk::JsonParam(encode_para_2)); */
/*     graph.Run(); */
/*     BMF_CPP_FILE_CHECK( */
/*         output_file, */
/*         "../transcode/unmuxed_output.mp4|240|320|7.574233|MOV,MP4,M4A,3GP,3G2,MJ2|404941|384340|h264|{\"fps\":
 * \"29.97002997\"}" */
/*     ); */
/* } */
