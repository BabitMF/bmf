#include "builder.hpp"
#include "bmf_nlohmann/json.hpp"
#include "graph_config.h"
#include <fstream>
#include <iostream>
#include <filesystem>

#include "cpp_test_helper.h"

TEST(cpp_fileconfig, run_by_config) {
    std::string output_file = "../files/out.mp4";
    BMF_CPP_FILE_REMOVE(output_file);

    bmf_nlohmann::json graph_para = {
        {"dump_graph", 1}
    };
    // auto graph = bmf::builder::Graph(bmf::builder::NormalMode, bmf_sdk::JsonParam(graph_para));
    std::string filepath = "../../example/run_by_config/config.json";
    // nlohmann::json graph_json;
    // std::ifstream gs(filepath);
    // gs >> graph_json;
    auto graph = bmf::BMFGraph(filepath, true, true);
    graph.start();
    graph.close();
    BMF_CPP_FILE_CHECK(
        output_file, 
        "../files/out.mp4|240|320|7.617000|MOV,MP4,M4A,3GP,3G2,MJ2|317620|302414|h264|{\"fps\": \"30.0662251656\"}"
    );
}
