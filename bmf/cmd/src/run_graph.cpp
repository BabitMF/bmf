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
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <bmf/sdk/compat/path.h>
#ifdef BMF_USE_MEDIACODEC
#include <jni.h>
#endif

#include "common.h"
#include "graph_config.h"
#include "graph.h"
#include "module_factory.h"
#include "optimizer.h"

#include "connector.hpp"

USE_BMF_ENGINE_NS
USE_BMF_SDK_NS

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "Invalid arguments!" << std::endl
                  << "Please use: run_bmf_graph /path_to_graph_config.json"
                  << std::endl;
        exit(EXIT_FAILURE);
    }

    // Graph config input
    const std::string filepath = argv[1];

    // Input file validation
    if (!fs::exists(filepath)) {
        std::cerr << "File does not exists!" << std::endl;
        exit(EXIT_FAILURE);
    }

    // Load file
    std::cout << "Loading graph config from " << filepath << std::endl;
    nlohmann::json graph_json;
    std::ifstream gs(filepath);
    gs >> graph_json;
    GraphConfig graph_config(graph_json);

    // Perform additional validations on input file paths
    std::string input_path_key = "input_path";
    for (auto node : graph_config.get_nodes()) {
        if (node.get_option().has_key(input_path_key)) {
            std::string input_path;
            node.get_option().get_string(input_path_key, input_path);
            if (!fs::exists(input_path)) {
                std::cerr << "Input file path " << input_path
                          << " does not exists!" << std::endl;
                exit(EXIT_FAILURE);
            }
        }
    }

    // Run the graph
    auto graph = bmf::BMFGraph(filepath, true, true);
    graph.start();
    graph.close();

    return 0;
}
#ifdef BMF_USE_MEDIACODEC
extern "C" JNIEXPORT void InitializeSignalChain() {}

extern "C" JNIEXPORT void ClaimSignalChain() {}

extern "C" JNIEXPORT void UnclaimSignalChain() {}

extern "C" JNIEXPORT void InvokeUserSignalHandler() {}

extern "C" JNIEXPORT void EnsureFrontOfChain() {}

extern "C" JNIEXPORT void AddSpecialSignalHandlerFn() {}

extern "C" JNIEXPORT void RemoveSpecialSignalHandlerFn() {}
#endif
