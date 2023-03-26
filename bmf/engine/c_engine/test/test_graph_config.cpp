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
#include "../include/common.h"
#include "../include/graph_config.h"

#include "gtest/gtest.h"

#include <fstream>

USE_BMF_ENGINE_NS
USE_BMF_SDK_NS
TEST(graph_config, graph_config) {
    bmf_nlohmann::json graph_json;
    std::string config_file = "../../../output/example/run_by_config/config.json";
    std::ifstream gs(config_file);
    gs >> graph_json;
    GraphConfig graph_config(graph_json);
    EXPECT_EQ(graph_config.get_nodes()[0].get_module_info().module_name, "c_ffmpeg_decoder");
}