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

#ifndef BMF_GRAPH_H
#define BMF_GRAPH_H

#include <bmf/sdk/common.h>
#include <bmf/sdk/module.h>
#include "graph_config.h"
#include "scheduler.h"
#include "input_stream.h"
#include "callback_layer.h"

#include "../../connector/include/running_info.h"

#include <cmath>
#include <memory>

BEGIN_BMF_ENGINE_NS
    USE_BMF_SDK_NS
    // TODO: unused
    enum class GraphState {
        GRAPH_NOT_RUNNING = 1,
        GRAPH_RUNNING = 2,
        GRAPH_PAUSE = 3,
        GRAPH_CANCEL = 4,
        GRAPH_TERMINATING = 5,
        GRAPH_TERMINATED = 6,
        GRAPH_PAUSED = 7
    };

    class GraphInputStream {
    public:
        void set_manager(std::shared_ptr<OutputStreamManager> &manager);

        void add_packet(Packet &packet);

        std::shared_ptr<OutputStreamManager> manager_;

    };

    class GraphOutputStream {
    public:
        void set_manager(std::shared_ptr<InputStreamManager> &input_manager);

        void set_node();

        void poll_packet(Packet &packet, bool block = true);

        std::shared_ptr<InputStreamManager> input_manager_;
    };

    class Graph {

    public:
        Graph(GraphConfig graph_config, std::map<int, std::shared_ptr<Module> > pre_modules,
              std::map<int, std::shared_ptr<ModuleCallbackLayer> > callback_bindings);

        ~Graph();
        
        void init(GraphConfig graph_config, std::map<int, std::shared_ptr<Module> > &pre_modules,
                  std::map<int, std::shared_ptr<ModuleCallbackLayer> > &callback_bindings);

//    Graph(std::string graph_config_file);
        int get_hungry_check_func(std::shared_ptr<Node> &root_node, int output_idx, std::shared_ptr<Node> &curr_node);

        int get_hungry_check_func_for_sources();

        int init_nodes();

        int init_input_streams();

        int add_all_mirrors_for_output_stream(std::shared_ptr<OutputStream> &stream);

        int find_orphan_input_streams();

        int start();

        int update(GraphConfig update_config);

        bool all_nodes_done();

        int get_scheduler();

        // timeout in ms
        void pause_running(double_t timeout = -1);

        void resume_running();

        int close();

        int force_close();

        int add_input_stream_packet(std::string const &stream_name, Packet &packet, bool block = false);

        Packet poll_output_stream_packet(std::string const &stream_name, bool block = true);

        int add_eos_packet(std::string const &stream_name);

        int get_node(int node_id, std::shared_ptr<Node> &node);

        void quit_gracefully();

        void print_node_info_pretty();

        bmf::GraphRunningInfo status();

    private:
        // visible to monitor.
        friend class RunningInfoCollector;

        bool paused_ = false;
        bool server_mode_ = false;
//    struct sigaction term_,intrpt_;
        BmfMode mode_;
        GraphConfig graph_config_;
        std::map<int, std::shared_ptr<Module> > pre_modules_;
        std::map<int, std::shared_ptr<ModuleCallbackLayer> > callback_bindings_;
        int scheduler_count_;
        std::shared_ptr<Scheduler> scheduler_;
        std::map<std::string, std::shared_ptr<GraphInputStream> > input_streams_;
        std::map<std::string, std::shared_ptr<GraphOutputStream> > output_streams_;
        std::map<int, std::shared_ptr<Node> > nodes_;
        std::vector<std::shared_ptr<Node> > source_nodes_;
//    std::map<std::string, OutputStreams> output_streams_;
//    std::map<std::string, InputStreams> input_streams_;
        std::vector<std::shared_ptr<InputStream>> orphan_streams_;
        bool py_init_flag_ = false;
        std::condition_variable cond_close_;
        std::mutex con_var_mutex_;
        int32_t closed_count_ = 0;
        bool exception_from_scheduler_ = false;
    };

END_BMF_ENGINE_NS
#endif //BMF_GRAPH_H
