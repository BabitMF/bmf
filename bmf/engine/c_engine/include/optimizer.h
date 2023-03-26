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

#ifndef BMF_ENGINE_OPTIMIZER_H
#define BMF_ENGINE_OPTIMIZER_H

#include <bmf/sdk/common.h>
#include <bmf/sdk/log.h>
#include <bmf/sdk/module.h>
#include "../include/graph_config.h"
#include "../include/module_factory.h"
#include <bmf_nlohmann/json_fwd.hpp>
#include <iostream>
#include <string>
#include <list>
#include <map>
#include <fstream>
#include <sstream>
#include <unordered_set>
#include <unordered_map>

using json = bmf_nlohmann::json;

#define Py_ADD_GIL PyGILState_STATE gstate = PyGILState_Ensure();{
#define Py_RELEASE_GIL }PyGILState_Release(gstate);

BEGIN_BMF_ENGINE_NS
    USE_BMF_SDK_NS

    namespace Optimizer {
        class Neighbour {
        public:
            StreamConfig root_stream;
            NodeConfig node;
        };

        void convert_filter_para(NodeConfig &node);
        void convert_filter_para_for_graph(std::vector<NodeConfig> &nodes);
        int find_merged_link(json &links, StreamConfig stream);
        void replace_stream_name_with_id(NodeConfig &node);
        void replace_stream_name_for_graph(std::vector<NodeConfig> &nodes);
        void merge_two_nodes(NodeConfig &n1, NodeConfig &n2);
        NodeConfig merge_ffmpeg_filter_nodes(std::vector<NodeConfig> &merge_nodes);
        std::vector<Neighbour> find_all_neighbours(std::vector<NodeConfig> opt_nodes, NodeConfig merged_node);
        StreamConfig has_circle(std::vector<NodeConfig> opt_nodes, NodeConfig merged_node, std::map<int, bool> &rec_stack);
        StreamConfig find_first_circle_node(std::vector<NodeConfig> opt_nodes, NodeConfig merged_node);
        void optimize(std::vector<NodeConfig> &nodes);
        void merge_subgraph(GraphConfig &main_config, GraphConfig &sub_config, int sub_node_id);
        void subgraph_preprocess(GraphConfig &main_graph_config, std::map<int, std::shared_ptr<Module> > &premodules);
        void dump_graph(GraphConfig graph_config, bool merged);
    }

END_BMF_ENGINE_NS

#endif //BMF_ENGINE_OPTIMIZER_H
