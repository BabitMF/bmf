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

#ifndef BMF_NODE_H
#define BMF_NODE_H

#include <bmf/sdk/common.h>
#include <bmf/sdk/task.h>
#include <bmf/sdk/module.h>
#include <bmf/sdk/exception_factory.h>
#include <bmf/sdk/error_define.h>

#include "module_factory.h"
#include "graph_config.h"
#include "input_stream_manager.h"
#include "output_stream_manager.h"
#include "callback_layer.h"



BEGIN_BMF_ENGINE_NS
    USE_BMF_SDK_NS
    enum class NodeState {
        NOT_INITED = 1,
        RUNNING = 2,
        PENDING = 3,
        CLOSED = 4,
        PAUSE_DONE = 5
    };

    class Node;

    class NodeCallBack {
    public:
        std::function<void(Task &)> scheduler_cb;
        std::function<void(int, int)> clear_cb;
        std::function<void(int, bool)> throttled_cb;
        std::function<void(int, bool)> sched_required;
        std::function<int(int, std::shared_ptr<Node> &)> get_node;
    };

    class Node {
    public:
        Node(int node_id, NodeConfig &node_config, NodeCallBack &callback,
             std::shared_ptr<Module> pre_allocated_modules, BmfMode mode,
             std::shared_ptr<ModuleCallbackLayer> callbacks);

        int process_node(Task &task);

        bool schedule_node();

        void wait_paused();

        bool is_source();

        void set_source(bool flag);

        int inc_pending_task();

        int dec_pending_task();

        int close();

        bool is_closed();

        int reset();

        void check_node_pending();

        int need_force_close();

        int need_opt_reset(JsonParam reset_opt);

        int all_downstream_nodes_closed();

        bool too_many_tasks_pending();

        bool any_of_downstream_full();

        bool is_hungry();

        void register_hungry_check_func(int input_idx, std::function<bool()> &func);

        void get_hungry_check_func(int input_idx, std::vector<std::function<bool()> > &funcs);

        int get_id();

        std::string get_alias();

        std::string get_action();

        int64_t get_source_timestamp();

        bool any_of_input_queue_full();

        bool all_input_queue_empty();

        void set_scheduler_queue_id(int scheduler_queue_id);

        int get_scheduler_queue_id();

        int get_output_streams(std::map<int, std::shared_ptr<OutputStream>> &output_streams);

        int get_input_stream_manager(std::shared_ptr<InputStreamManager> &);

        int get_output_stream_manager(std::shared_ptr<OutputStreamManager> &);

        int get_input_streams(std::map<int, std::shared_ptr<InputStream> > &input_streams);

        std::string get_type();

        std::string get_status();

        void set_status(NodeState state);

        void set_outputstream_updated(bool update);

        int get_schedule_attempt_cnt();

        int get_schedule_success_cnt();

        int64_t get_last_timestamp();

        long long dur;

        bool wait_pause_;

        int64_t pre_sched_num_ = 0;

        std::mutex sched_mutex_;

        BmfMode mode_;
    private:
        // visible to monitor
        friend class RunningInfoCollector;

        int id_;
        std::string node_name_;
        std::string module_name_;
        int scheduler_queue_id_;
        long long task_processed_cnt_;
        bool is_premodule_;
        NodeConfig node_config_;
        std::string type_;
        bool is_source_;
        int pending_tasks_ = 0;
        int max_pending_tasks_ = 5;
        uint32_t queue_size_limit_ = 5;
        bool infinity_node_ = false;
        bool node_output_updated_ = false;
        ModuleInfo module_info_;
        std::shared_ptr<Module> module_;
        std::shared_ptr<ModuleCallbackLayer> module_callbacks_;
        std::shared_ptr<InputStreamManager> input_stream_manager_;
        std::shared_ptr<OutputStreamManager> output_stream_manager_;
        bool force_close_;
        NodeState state_;
        JsonParam reset_option_;
        bool need_opt_reset_;
        std::mutex opt_reset_mutex_;
        int schedule_node_cnt_;
        int schedule_node_success_cnt_;
        int64_t source_timestamp_;
        int64_t last_timestamp_;
        mutable std::recursive_mutex mutex_;
        mutable std::mutex pause_mutex_;
        std::condition_variable pause_event_;
        NodeCallBack callback_;
        std::map<int, std::vector<std::function<bool()> > > hungry_check_func_;
    };
END_BMF_ENGINE_NS

#endif //BMF_NODE_H
