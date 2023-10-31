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
#include <unistd.h>
#include "nlohmann/json.hpp"

#include "cpp_test_helper.h"

TEST(cpp_set_option, set_option) {
    std::string graph_json = "graph.json";
    BMF_CPP_FILE_REMOVE(graph_json);

    std::string output_file = "./output.mp4";
    BMF_CPP_FILE_REMOVE(output_file);

    nlohmann::json graph_para = {{"dump_graph", 0}};
    auto graph = bmf::builder::Graph(bmf::builder::NormalMode,
                                     bmf_sdk::JsonParam(graph_para));

    nlohmann::json decode_para = {
        {"input_path", "../../files/big_bunny_10s_30fps.mp4"}};
    auto video = graph.Decode(bmf_sdk::JsonParam(decode_para));

#ifdef WIN32
    auto video_2 =
        graph.Module({video["video"]}, "copy_module", bmf::builder::CPP,
                     bmf_sdk::JsonParam(), "CopyModule",
                     "../lib/copy_module.dll", "copy_module:CopyModule");
#else
    auto video_2 =
        graph.Module({video["video"]}, "copy_module", bmf::builder::CPP,
                     bmf_sdk::JsonParam(), "CopyModule",
                     "../lib/libcopy_module.so", "copy_module:CopyModule");
#endif
    nlohmann::json encode_para = {
        {"output_path", output_file},
        {"video_params", {{"vsync", "vfr"}, {"max_fr", 60}}},
        {"audio_params", {{"codec", "aac"}}}};
    graph.Encode(video_2, video["audio"], bmf_sdk::JsonParam(encode_para));

    nlohmann::json option_patch = {{"dump_graph", 1}};
    graph.SetOption(bmf_sdk::JsonParam(option_patch));
    graph.Run(false);

#ifdef _WIN32
    EXPECT_EQ(true, _access(graph_json.c_str(), 0) == 0);
#else
    EXPECT_EQ(true, access(graph_json.c_str(), F_OK) == 0);
#endif
}
