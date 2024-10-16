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
#include "builder.hpp"
#include "nlohmann/json.hpp"

void test_task() {
    int scheduler_cnt = 0;
    int dist_nums = 3;
    auto graph = bmf::builder::Graph(bmf::builder::NormalMode);
    // decoder init 
    nlohmann::json decode_para = {
        {"input_path", "../../../files/big_bunny_10s_30fps.mp4"}
    };
    auto video = graph.Decode(bmf_sdk::JsonParam(decode_para), "", scheduler_cnt++);
    /* distributed ndoe init */
    nlohmann::json node_para = {
        {"dist_nums", dist_nums}
    };
    auto video_copied = 
        graph.Module({video["video"]}, 
                    "copy_module", bmf::builder::CPP,
                    bmf_sdk::JsonParam(node_para), "CopyModule",
                    "./libcopy_module.so", "copy_module:CopyModule",
                    bmf::builder::Immediate, scheduler_cnt++);

    // encoder init 
    nlohmann::json encode_para = {
        {"output_path", "./rgb2video.mp4"},
    };
    graph.Encode(video_copied, video["audio"], bmf_sdk::JsonParam(encode_para), "", scheduler_cnt++);

    nlohmann::json graph_para = {{"dump_graph", 1}};
    graph.SetOption(bmf_sdk::JsonParam(graph_para));
    graph.Run();
    // std::cout << graph.Dump() << std::endl;
}

int main() {
    // task();
    // mock_task();
    test_task();
}
