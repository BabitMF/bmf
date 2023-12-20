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
#include <bmf/sdk/compat/path.h>
#ifdef BMF_USE_MEDIACODEC
#include <jni.h>
#endif

#include "common.h"

#include "connector.hpp"

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
