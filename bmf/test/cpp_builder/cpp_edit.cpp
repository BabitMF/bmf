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

TEST(cpp_edit, edit_concat) {
    std::string output_file = "./video_concat.mp4";
    BMF_CPP_FILE_REMOVE(output_file);

    nlohmann::json graph_para = {{"dump_graph", 0}};
    auto graph = bmf::builder::Graph(bmf::builder::NormalMode,
                                     bmf_sdk::JsonParam(graph_para));

    nlohmann::json decode_para = {
        {"input_path", "../../files/big_bunny_10s_30fps.mp4"}};

    std::vector<bmf::builder::Stream> video_concat_streams;
    std::vector<bmf::builder::Stream> video_transit_streams;
    std::vector<bmf::builder::Stream> audio_concat_streams;

    for (int i = 0; i < 3; i++) {
        auto video = graph.Decode(bmf_sdk::JsonParam(decode_para));

        // Process video streams
        auto video_stream = video["video"].Scale("1280:720");

        if (i < 2) {
            auto split_stream = video_stream.Split("");
            auto concat_stream = split_stream[0]
                                     .Trim("start=0:duration=5")
                                     .Setpts("PTS-STARTPTS");

            auto transition_stream = split_stream[1]
                                         .Trim("start=5:duration=2")
                                         .Setpts("PTS-STARTPTS")
                                         .Scale("200:200");

            video_transit_streams.push_back(transition_stream);
            video_concat_streams.push_back(concat_stream);
        } else {
            auto concat_stream =
                video_stream.Trim("start=0:duration=5").Setpts("PTS-STARTPTS");

            video_concat_streams.push_back(concat_stream);
        }

        if (i > 0) {
            auto concat_stream = video_concat_streams[i].Overlay(
                {video_transit_streams[i - 1]}, "repeatlast=0");
            video_concat_streams.pop_back();
            video_concat_streams.push_back(concat_stream);
        }

        // Process audio streams
        auto audio_stream = video["audio"]
                                .Atrim("start=0:duration=5")
                                .Asetpts("PTS-STARTPTS")
                                .Afade("t=in:st=0:d=2")
                                .Afade("t=out:st=5:d=2");

        audio_concat_streams.push_back(audio_stream);
    }

    auto concat_video = graph.Concat(video_concat_streams, "n=3:v=1:a=0");
    auto concat_audio = graph.Concat(audio_concat_streams, "n=3:v=0:a=1");

    nlohmann::json encode_para = {
        {"output_path", output_file},
        {"video_params", {{"width", 1280}, {"height", 720}}}};
    graph.Encode(concat_video, concat_audio, bmf_sdk::JsonParam(encode_para));

    graph.Run();

    BMF_CPP_FILE_CHECK(output_file, "../edit/"
                                    "video_concat.mp4|720|1280|15.022000|MOV,"
                                    "MP4,M4A,3GP,3G2,MJ2|385322|722480|h264|{"
                                    "\"fps\": \"30.0166759311\"}");
}

TEST(cpp_edit, edit_overlay) {
    std::string output_file = "./overlays.mp4";
    BMF_CPP_FILE_REMOVE(output_file);

    nlohmann::json graph_para = {{"dump_graph", 0}};
    auto graph = bmf::builder::Graph(bmf::builder::NormalMode,
                                     bmf_sdk::JsonParam(graph_para));

    nlohmann::json decode_para = {
        {"input_path", "../../files/big_bunny_10s_30fps.mp4"}};
    auto video = graph.Decode(bmf_sdk::JsonParam(decode_para));

    nlohmann::json logoPara = {
        {"input_path", "../../files/xigua_prefix_logo_x.mov"}};
    auto logo = graph.Decode(bmf_sdk::JsonParam(logoPara));

    auto output_stream = video["video"]
                             .Scale("1280:720")
                             .Trim("start=0:duration=7")
                             .Setpts("PTS-STARTPTS");

    auto overlay = logo["video"]
                       .Scale("300:200")
                       .Loop("loop=0:size=10000")
                       .Setpts("PTS+0/TB");

    nlohmann::json encode_para = {
        {"output_path", output_file},
        {"video_params", {{"width", 640}, {"height", 480}, {"codec", "h264"}}}};

    output_stream[0]
        .Overlay({overlay}, "x=if(between(t,0,7),0,NAN):y=if(between(t,0,7),0,"
                            "NAN):repeatlast=1")
        .EncodeAsVideo(bmf_sdk::JsonParam(encode_para));

    graph.Run();

    BMF_CPP_FILE_CHECK(output_file, "../edit/"
                                    "overlays.mp4|480|640|6.984000|MOV,MP4,M4A,"
                                    "3GP,3G2,MJ2|132817|116215|h264|{\"fps\": "
                                    "\"30.0715990453\"}");
}

/* TEST(cpp_edit, edit_audio_mix) { */
/*     std::string output_file = "./audio_mix.mp4"; */
/*     BMF_CPP_FILE_REMOVE(output_file); */

/*     nlohmann::json graph_para = { */
/*         {"dump_graph", 0} */
/*     }; */
/*     auto graph = bmf::builder::Graph(bmf::builder::NormalMode,
 * bmf_sdk::JsonParam(graph_para)); */

/*     nlohmann::json decode_para = { */
/*         {"input_path", "../../files/big_bunny_10s_30fps.mp4"} */
/*     }; */
/*     auto video = graph.Decode(bmf_sdk::JsonParam(decode_para)); */
/*     auto video2 = graph.Decode(bmf_sdk::JsonParam(decode_para)); */
/*     nlohmann::json module_para = { */
/*         {"audios", { */
/*             { */
/*                 {"start", 1}, */
/*                 {"duration", 5} */
/*             }, */
/*             { */
/*                 {"start", 0}, */
/*                 {"duration", 5} */
/*             } */
/*         } */
/*         } */
/*     }; */

/*     auto audio_stream = graph.Module({video["audio"], video2["audio"]},
 * "audio_mix", bmf::builder::Python, bmf_sdk::JsonParam(module_para), */
/*             "MyModule", "../../example/edit", "audio_mix:audio_mix" */
/*         ); */

/*     nlohmann::json encode_para = { */
/*         {"output_path", output_file}, */
/*         {"video_params", { */
/*             {"width", 640}, */
/*             {"height", 480} */
/*         }} */
/*     }; */
/*     video["video"].EncodeAsVideo(audio_stream,
 * bmf_sdk::JsonParam(encode_para)); */
/*     std::cout << graph.Dump() << std::endl; */
/*     graph.Run(); */

/*     BMF_CPP_FILE_CHECK( */
/*         output_file, */
/*         "../edit/audio_mix.mp4|480|640|7.550000|MOV,MP4,M4A,3GP,3G2,MJ2|1143753|1079417|h264|{\"fps\":
 * \"30.0662251656\"}" */
/*     ); */
/* } */
