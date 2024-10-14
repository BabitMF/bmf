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
    /* decoder init */
    nlohmann::json decode_para = {
        {"input_path", "../../files/big_bunny_10s_30fps.mp4"}
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

    /* encoder init */
    nlohmann::json encode_para = {
        {"output_path", "./rgb2video.mp4"},
    };
    graph.Encode(video_copied, video["audio"], bmf_sdk::JsonParam(encode_para), "", scheduler_cnt++);

    nlohmann::json graph_para = {{"dump_graph", 1}};
    graph.SetOption(bmf_sdk::JsonParam(graph_para));
    graph.Run();
    // std::cout << graph.Dump() << std::endl;
}

void task() {
    size_t multi_nums = 3;
    int scheduler_cnt = 0;
    auto graph = bmf::builder::Graph(bmf::builder::NormalMode);

    nlohmann::json decode_para = {
        {"input_path", "../../files/big_bunny_10s_30fps.mp4"}};
    auto video = graph.Decode(bmf_sdk::JsonParam(decode_para), "", 0);

    auto video_split = 
        graph.Module({video["video"]}, 
            "assemble_module", bmf::builder::CPP,
            bmf_sdk::JsonParam(), "SplitModule",
            "/root/workspace/bmf_OSPP/output/bmf/cpp_modules/Module_split/libsplit.so", "split_module:SplitModule",
            bmf::builder::Immediate, 0);

    std::vector<bmf::builder::Node> video_copied;
    for (size_t i = 0; i < multi_nums; i++) {
        video_copied.push_back(
            graph.Module({video["video"]}, 
                        "copy_module", bmf::builder::CPP,
                        bmf_sdk::JsonParam(), "CopyModule",
                        "./libcopy_module.so", "copy_module:CopyModule",
                        bmf::builder::Immediate, 1)
        );
    }

    video_copied.push_back(
        graph.Module({video_copied[0], video_copied[1], video_copied[2]}, 
                    "assemble_module", bmf::builder::CPP,
                    bmf_sdk::JsonParam(), "AssembleModule",
                    "/root/workspace/bmf_OSPP/output/bmf/cpp_modules/Module_assemble/libassemble.so", "assemble_module:AssembleModule",
                    bmf::builder::Immediate, 0)
    );

    nlohmann::json encode_para = {
        {"output_path", "./rgb2video.mp4"},
    };
    graph.Module({video_copied[3]},
            "c_ffmpeg_encoder", bmf::builder::CPP,
            bmf_sdk::JsonParam(encode_para), "",
            "", "",
            bmf::builder::Immediate, 2);

    nlohmann::json graph_para = {{"dump_graph", 1},
                                 {"scheduler_count", 6}};
    graph.SetOption(bmf_sdk::JsonParam(graph_para));
    graph.Run();
}

void mock_task() {
    size_t multi_nums = 3;
    int scheduler_cnt = 0;
    auto graph = bmf::builder::Graph(bmf::builder::NormalMode);

    nlohmann::json decode_para = {
        {"input_path", "../../files/big_bunny_10s_30fps.mp4"}};
    auto video = graph.Decode(bmf_sdk::JsonParam(decode_para), "", scheduler_cnt++);
    // auto video = 
    //     graph.Module({}, 
    //                 "c_ffmpeg_decoder", bmf::builder::CPP,
    //                 bmf_sdk::JsonParam(decode_para), "",
    //                 "", "",
    //                 bmf::builder::Immediate, scheduler_cnt++);
    // auto video = graph.NewNode("", bmf_sdk::JsonParam(decode_para), {}, "c_ffmpeg_decoder", bmf::builder::CPP, "", "",
    //                             bmf::builder::Immediate, scheduler_cnt);
    std::vector<bmf::builder::Node> video_copied;
    for (size_t i = 0; i < multi_nums; i++) {
        /* pass through module */
        // video_copied.push_back(
        //     graph.Module({video["video"]}, 
        //                 "pass_through", bmf::builder::CPP,
        //                 bmf_sdk::JsonParam(), "PassThroughModule",
        //                 "/root/workspace/bmf_OSPP/output/bmf/cpp_modules/Module_pass_through/libpass_through.so", "pass_through:PassThroughModule",
        //                 bmf::builder::Immediate, scheduler_cnt++)
        // );
        /* c_copy_module */
        video_copied.push_back(
            graph.Module({video["video"]}, 
                        "copy_module", bmf::builder::CPP,
                        bmf_sdk::JsonParam(), "CopyModule",
                        "./libcopy_module.so", "copy_module:CopyModule",
                        bmf::builder::Immediate, scheduler_cnt++)
        );
        // auto video_copied =
        //     graph.Module({video["video"]}, 
        //                 "copy_module", bmf::builder::CPP,
        //                 bmf_sdk::JsonParam(), "CopyModule",
        //                 "./libcopy_module.so", "copy_module:CopyModule",
        //                 bmf::builder::Immediate, scheduler_cnt++);
    }
    
    /* single -> multi -> multi(this)  */
    // for (size_t i = 0; i < multi_nums; i++) {
    //     graph.Module({video_copied[i]}, 
    //                 "copy_module", bmf::builder::CPP,
    //                 bmf_sdk::JsonParam(), "CopyModule",
    //                 "./libcopy_module.so", "copy_module:CopyModule",
    //                 bmf::builder::Immediate, scheduler_cnt++);
    // }
    
    /* single -> multi -> single(this) -> single  */
    /* pass through module */
    video_copied.push_back(
        graph.Module({video_copied[0], video_copied[1], video_copied[2]}, 
                    "pass_through", bmf::builder::CPP,
                    bmf_sdk::JsonParam(), "PassThroughModule",
                    "/root/workspace/bmf_OSPP/output/bmf/cpp_modules/Module_pass_through/libpass_through.so", "pass_through:PassThroughModule",
                    bmf::builder::Immediate, scheduler_cnt++)
    );
    /* c_copy_module */
    // video_copied.push_back(
    //     graph.Module({video_copied[0], video_copied[1], video_copied[2]}, 
    //                 "copy_module", bmf::builder::CPP,
    //                 bmf_sdk::JsonParam(), "CopyModule",
    //                 "./libcopy_module.so", "copy_module:CopyModule",
    //                 bmf::builder::Immediate, scheduler_cnt++)
    // );
    /* three input stream, three encoder */
    // std::vector<nlohmann::json> encode_para = {
    //     {{"output_path", "./rgb2video_1.mp4"}},
    //     {{"output_path", "./rgb2video_2.mp4"}},
    //     {{"output_path", "./rgb2video_3.mp4"}}
    // };
    // for (size_t i = 0; i < 3; i++) {
    //     graph.Module({video_copied[i]},
    //         "c_ffmpeg_encoder", bmf::builder::CPP,
    //         bmf_sdk::JsonParam(encode_para[i]), "",
    //         "", "",
    //         bmf::builder::Immediate, scheduler_cnt++);
    // }
    /* one input stream, one encoder */
    // nlohmann::json encode_para = {
    //     {"output_path", "./rgb2video_1.mp4"},
    // };
    // graph.Module({video_copied[0]},
    //         "c_ffmpeg_encoder", bmf::builder::CPP,
    //         bmf_sdk::JsonParam(encode_para), "",
    //         "", "",
    //         bmf::builder::Immediate, scheduler_cnt++);
    // graph.Encode(video_copied[0],
    //              bmf_sdk::JsonParam(encode_para),
    //              "", scheduler_cnt++);
    /* three input stream, one encoder */
    // nlohmann::json encode_para = {
    //     {"output_path", "./rgb2video.mp4"},
    // };
    // graph.Module({video_copied[0], video_copied[1], video_copied[2]},
    //         "c_ffmpeg_encoder", bmf::builder::CPP,
    //         bmf_sdk::JsonParam(encode_para), "",
    //         "", "",
    //         bmf::builder::Immediate, scheduler_cnt++);
    /* 1 -> 3 -> 1 -> 1 */
    nlohmann::json encode_para = {
        {"output_path", "./rgb2video.mp4"},
    };
    graph.Module({video_copied[3]},
            "c_ffmpeg_encoder", bmf::builder::CPP,
            bmf_sdk::JsonParam(encode_para), "",
            "", "",
            bmf::builder::Immediate, scheduler_cnt++);

    nlohmann::json graph_para = {{"dump_graph", 1},
                                 {"scheduler_count", scheduler_cnt}};
    graph.SetOption(bmf_sdk::JsonParam(graph_para));
    graph.Run();
}

int main() {
    // task();
    // mock_task();
    test_task();
}
