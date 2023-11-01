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

TEST(cpp_modules, module_python) {
    std::string output_file = "./output.mp4";
    BMF_CPP_FILE_REMOVE(output_file);

    nlohmann::json graph_para = {{"dump_graph", 1}};
    auto graph = bmf::builder::Graph(bmf::builder::NormalMode,
                                     bmf_sdk::JsonParam(graph_para));

    nlohmann::json decode_para = {
        {"input_path", "../../files/big_bunny_10s_30fps.mp4"}};
    auto video = graph.Decode(bmf_sdk::JsonParam(decode_para));

    nlohmann::json encode_para = {{"output_path", output_file}};

    graph
        .Module({video["video"]}, "python_copy_module", bmf::builder::Python,
                bmf_sdk::JsonParam())
        .EncodeAsVideo(video["audio"], bmf_sdk::JsonParam(encode_para));

    graph.Run();

    BMF_CPP_FILE_CHECK(output_file, "|1080|1920|10.008|MOV,MP4,M4A,3GP,3G2,MJ2|"
                                    "1918874|2400512|h264|{\"fps\": "
                                    "\"30.0662251656\"}");
}

TEST(cpp_modules, module_cpp) {
    std::string output_file = "./output.mp4";
    BMF_CPP_FILE_REMOVE(output_file);

    nlohmann::json graph_para = {{"dump_graph", 1}};
    auto graph = bmf::builder::Graph(bmf::builder::NormalMode,
                                     bmf_sdk::JsonParam(graph_para));

    nlohmann::json decode_para = {
        {"input_path", "../../files/big_bunny_10s_30fps.mp4"}};
    auto video = graph.Decode(bmf_sdk::JsonParam(decode_para));

    auto video_2 =
        graph.Module({video["video"]}, "copy_module", bmf::builder::CPP,
                     bmf_sdk::JsonParam(), "CopyModule",
                     "../lib/libcopy_module.so", "copy_module:CopyModule");

    nlohmann::json encode_para = {
        {"output_path", output_file},
        {"video_params", {{"vsync", "vfr"}, {"max_fr", 60}}},
        {"audio_params", {{"codec", "aac"}}}};
    graph.Encode(video_2, video["audio"], bmf_sdk::JsonParam(encode_para));

    graph.Run();

    BMF_CPP_FILE_CHECK(output_file, "../c_module/"
                                    "output.mp4|1080|1920|10.008000|MOV,MP4,"
                                    "M4A,3GP,3G2,MJ2|1918880|2400520|h264|{"
                                    "\"fps\": \"30.0662251656\"}");
}

TEST(cpp_modules, audio_python_module) {
    std::string output_file = "./audio_python_module";
    BMF_CPP_FILE_REMOVE(output_file);

    nlohmann::json graph_para = {{"dump_graph", 1}};
    auto graph = bmf::builder::Graph(bmf::builder::NormalMode,
                                     bmf_sdk::JsonParam(graph_para));

    nlohmann::json decode_para = {
        {"input_path", "../../files/big_bunny_10s_30fps.mp4"}};
    auto audio = graph.Decode(bmf_sdk::JsonParam(decode_para))["audio"];

    nlohmann::json encode_para = {{"output_path", output_file}};

    auto audio_output =
        graph.Module({audio}, "python_copy_module", bmf::builder::Python,
                     bmf_sdk::JsonParam());
    graph.Encode(graph.NewPlaceholderStream(), audio_output,
                 bmf_sdk::JsonParam(encode_para));
    graph.Run();

    BMF_CPP_FILE_CHECK(output_file, "../audio_copy/"
                                    "audio_c_module.mp4|0|0|10.008000|MOV,MP4,"
                                    "M4A,3GP,3G2,MJ2|136031|166183||{}");
}

TEST(cpp_modules, test_exception_in_python_module) {
    std::string output_file = "./test_exception_in_python_module.mp4";

    nlohmann::json graph_para = {{"dump_graph", 1}};
    auto graph = bmf::builder::Graph(bmf::builder::NormalMode,
                                     bmf_sdk::JsonParam(graph_para));

    nlohmann::json decode_para = {
        {"input_path", "../../files/big_bunny_10s_30fps.mp4"}};
    auto audio = graph.Decode(bmf_sdk::JsonParam(decode_para))["audio"];

    nlohmann::json encode_para = {{"output_path", output_file}};
    nlohmann::json module_para = {{"exception", 1}};

    auto audio_output =
        graph.Module({audio}, "my_module", bmf::builder::Python,
                     bmf_sdk::JsonParam(module_para), "MyModule",
                     "../../example/customize_module", "my_module:my_module");
    try {
        graph.Encode(graph.NewPlaceholderStream(), audio_output,
                     bmf_sdk::JsonParam(encode_para));
        graph.Run();
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
    }
}
