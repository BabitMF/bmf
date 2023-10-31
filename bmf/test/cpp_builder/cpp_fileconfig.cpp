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
#include "graph_config.h"
#include <fstream>
#include <iostream>
#include <filesystem>

#include "cpp_test_helper.h"

TEST(cpp_fileconfig, run_by_config) {
    std::string output_file = "../../files/out.mp4";
    BMF_CPP_FILE_REMOVE(output_file);

    nlohmann::json graph_para = {{"dump_graph", 1}};
    // auto graph = bmf::builder::Graph(bmf::builder::NormalMode,
    // bmf_sdk::JsonParam(graph_para));
    std::string filepath = "../../test/run_by_config/config.json";
    // nlohmann::json graph_json;
    // std::ifstream gs(filepath);
    // gs >> graph_json;
    auto graph = bmf::BMFGraph(filepath, true, true);
    graph.start();
    graph.close();
    BMF_CPP_FILE_CHECK(output_file, "../../files/"
                                    "out.mp4|240|320|10.008|MOV,MP4,M4A,3GP,"
                                    "3G2,MJ2|175470|219513|h264|{\"fps\": "
                                    "\"30.0662251656\"}");
}
