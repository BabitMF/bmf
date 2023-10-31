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

void rgb2video() {
    nlohmann::json graph_para = {{"dump_graph", 1}};
    auto graph = bmf::builder::Graph(bmf::builder::NormalMode,
                                     bmf_sdk::JsonParam(graph_para));

    nlohmann::json decode_para = {
        {"input_path", "/opt/tiger/bmf/test/files/test_rgba_806x654.rgb"},
        {"s", "806:654"},
        {"pix_fmt", "rgba"}};
    auto stream = graph.Decode(bmf_sdk::JsonParam(decode_para));

    auto video_stream =
        stream["video"].FFMpegFilter({}, "loop", "loop=50:size=1");

    nlohmann::json encode_para = {
        {"output_path", "./rgb2video.mp4"},
    };

    // graph.Encode(video_stream, bmf_sdk::JsonParam(encode_para));

    graph.Encode(video_stream, stream["audio"],
                 bmf_sdk::JsonParam(encode_para));

    graph.Run();
}
