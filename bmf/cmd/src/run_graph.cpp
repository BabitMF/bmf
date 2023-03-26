#include <fstream>
#include <iostream>
#include <bmf_nlohmann/json.hpp>
#include <bmf/sdk/compat/path.h>

#include "common.h"
#include "graph_config.h"
#include "graph.h"
#include "module_factory.h"
#include "optimizer.h"

#include "connector.hpp"

USE_BMF_ENGINE_NS
USE_BMF_SDK_NS

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::cerr
            << "Invalid arguments!"
            << std::endl
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
    bmf_nlohmann::json graph_json;
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
                std::cerr
                    << "Input file path " << input_path 
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