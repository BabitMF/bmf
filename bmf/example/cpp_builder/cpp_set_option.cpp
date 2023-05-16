#include "builder.hpp"
#include <unistd.h>
#include "bmf_nlohmann/json.hpp"

#include "cpp_test_helper.h"

TEST(cpp_set_option, set_option) {
    std::string graph_json = "graph.json";
    BMF_CPP_FILE_REMOVE(graph_json);

    std::string output_file = "./output.mp4";
    BMF_CPP_FILE_REMOVE(output_file);

    bmf_nlohmann::json graph_para = {
        {"dump_graph", 0}
    };
    auto graph = bmf::builder::Graph(bmf::builder::NormalMode, bmf_sdk::JsonParam(graph_para));

    bmf_nlohmann::json decode_para = {
        {"input_path", "../../example/files/img.mp4"}
    };
    auto video = graph.Decode(bmf_sdk::JsonParam(decode_para));

    auto video_2 = graph.Module(
        {video["video"]}, 
        "copy_module", 
        bmf::builder::CPP, 
        bmf_sdk::JsonParam(), 
        "CopyModule", 
        "../lib/libcopy_module.so", 
        "copy_module:CopyModule"
    );

    bmf_nlohmann::json encode_para = {
        {"output_path", output_file},
        {"video_params", {
            {"vsync", "vfr"},
            {"max_fr", 60}
        }},
        {"audio_params", {
            {"codec", "aac"}
        }}
    };
    graph.Encode(video_2, video["audio"], bmf_sdk::JsonParam(encode_para));

    bmf_nlohmann::json option_patch = {
        {"dump_graph", 1}
    };
    graph.SetOption(bmf_sdk::JsonParam(option_patch));
    graph.Run(false);

    EXPECT_EQ(true, access(graph_json.c_str(), F_OK) == 0);
}
