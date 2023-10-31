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

TEST(cpp_premodule, premodule) {
    std::string output_file = "./output.mp4";
    BMF_CPP_FILE_REMOVE(output_file);

    nlohmann::json pre_module_option = {{"name", "analysis_SR"},
                                        {"para", "analysis_SR"}};
    auto pre_module = bmf::builder::GetModuleInstance(
        "analysis", pre_module_option.dump(), bmf::builder::Python,
        "../../test/pre_module", "analysis:analysis");

    for (int i = 0; i < 3; i++) {
        nlohmann::json graph_para = {{"dump_graph", 1}};
        nlohmann::json decode_para = {
            {"input_path", "../../files/big_bunny_10s_30fps.mp4"}};
        auto graph = bmf::builder::Graph(bmf::builder::NormalMode,
                                         bmf_sdk::JsonParam(graph_para));
        auto video = graph.Decode(bmf_sdk::JsonParam(decode_para));

        auto output = video["video"].Scale("320:240");
        auto analyzed =
            output.PythonModule({}, "analysis", bmf_sdk::JsonParam());
        analyzed.SetPreModule(pre_module);

        nlohmann::json encode_para = {
            {"output_path", output_file},
            {"video_params", {{"width", 300}, {"height", 200}}}};
        analyzed.EncodeAsVideo(bmf_sdk::JsonParam(encode_para));
        graph.Run();

        BMF_CPP_FILE_CHECK(output_file, "../pre_module/"
                                        "output.mp4|200|300|10.0|MOV,MP4,M4A,"
                                        "3GP,3G2,MJ2|62956|78695|h264|{\"fps\":"
                                        " \"30.0662251656\"}");
    }
}
