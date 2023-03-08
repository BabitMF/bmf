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

#ifndef BMF_INPUT_STREAM_MANAGER_H
#define BMF_INPUT_STREAM_MANAGER_H

#include "input_stream.h"
#include "graph_config.h"

#include <bmf/sdk/task.h>

#include <mutex>
#include <set>

BEGIN_BMF_ENGINE_NS
    USE_BMF_SDK_NS

    class Node;

    enum class NodeReadiness {
        NOT_READY = 1,
        READY_FOR_PROCESS = 2,
        READY_FOR_CLOSE = 3
    };

    class InputStreamManagerCallBack {
    public:
        std::function<void(Task &)> scheduler_cb = NULL;
        std::function<bool()> notify_cb = NULL;
        std::function<void(int, bool)> throttled_cb = NULL;
        std::function<void(int, bool)> sched_required = NULL;
        std::function<bool()> node_is_closed_cb = NULL;
        std::function<int(int, std::shared_ptr<Node> &)> get_node_cb = NULL;
    };

    class InputStreamManager {
    public:
        InputStreamManager(int node_id, std::vector<StreamConfig> &input_streams,
                           std::vector<int> &output_stream_id_list, uint32_t max_queue_size,
                           InputStreamManagerCallBack &callback);

        virtual std::string type() = 0;

        virtual NodeReadiness get_node_readiness(int64_t &min_timestamp) = 0;

        virtual bool fill_task_input(Task &task) = 0;

        bool get_stream(int stream_id, std::shared_ptr<InputStream> &input_stream);

        int add_stream(std::string name, int id);

        int remove_stream(int stream_id);

        int wait_on_stream_empty(int stream_id);

        Packet pop_next_packet(int stream_id, bool block = true);

        bool schedule_node();

        void add_packets(int stream_id, std::shared_ptr<SafeQueue<Packet> > packets);

        int add_upstream_nodes(int node_id);
        void remove_upstream_nodes(int node_id);
        bool find_upstream_nodes(int node_id);

        int node_id_;
        std::map<int, std::shared_ptr<InputStream> > input_streams_;
        InputStreamManagerCallBack callback_;
        std::vector<std::string> input_stream_names_;
        std::vector<int> stream_id_list_;
        std::vector<int> output_stream_id_list_;
        mutable std::mutex stream_mutex_;
        std::map<int, int> stream_done_;
        int max_id_;
        std::mutex mtx_;
        std::set<int> upstream_nodes_;
    };

    class DefaultInputManager : public InputStreamManager {
    public:
        DefaultInputManager(int node_id, std::vector<StreamConfig> &input_streams,
                            std::vector<int> &output_stream_id_list, uint32_t max_queue_size,
                            InputStreamManagerCallBack &callback);

        std::string type() override;

        NodeReadiness get_node_readiness(int64_t &min_timestamp) override;

        bool fill_task_input(Task &task) override;

    private:

    };

    class ImmediateInputStreamManager : public InputStreamManager {
    public:
        ImmediateInputStreamManager(int node_id, std::vector<StreamConfig> &input_streams,
                                    std::vector<int> &output_stream_id_list, uint32_t max_queue_size,
                                    InputStreamManagerCallBack &callback);

        std::string type() override;

        int64_t get_next_timestamp();

        NodeReadiness get_node_readiness(int64_t &min_timestamp) override;

        bool fill_task_input(Task &task) override;

//    void get_stream(std::string stream_id);
//    void pop_next_packet(std::string stream_id, bool block = true);

//    bool schedule_node();
        int64_t next_timestamp_;
    };

    class ServerInputStreamManager : public InputStreamManager {
    public:
        ServerInputStreamManager(int node_id, std::vector<StreamConfig> &input_streams,
                                 std::vector<int> &output_stream_id_list, uint32_t max_queue_size,
                                 InputStreamManagerCallBack &callback);

        std::string type() override;

        int64_t get_next_time_stamp();

        NodeReadiness get_node_readiness(int64_t &next_timestamp) override;

        int get_positive_stream_eof_number();

        void update_stream_eof();

        bool fill_task_input(Task &task) override;

    private:
        int64_t next_timestamp_;
        std::map<std::shared_ptr<InputStream>, int> stream_eof_;
    };

    class FrameSyncInputStreamManager : public InputStreamManager {
    public:
        FrameSyncInputStreamManager(int node_id, std::vector<StreamConfig> &input_streams,
                                    std::vector<int> &output_stream_id_list, uint32_t max_queue_size,
                                    InputStreamManagerCallBack &callback);

        std::string type() override;

        NodeReadiness get_node_readiness(int64_t &min_timestamp) override;

        bool fill_task_input(Task &task) override;

    private:
        int sync_level_;
        std::map<int, Packet> next_pkt_;
        std::map<int, Packet> curr_pkt_;
        std::map<int, std::shared_ptr<SafeQueue<Packet>>> pkt_ready_;
        bool frames_ready_;
        int64_t next_timestamp_;
        std::map<int, bool> have_next_;
        std::map<int, enum Timestamp> sync_frm_state_;
        std::map<int, int64_t> timestamp_;
        std::map<int, int64_t> timestamp_next_;
    };

    class ClockBasedSyncInputStreamManager : public InputStreamManager {
    public:
        ClockBasedSyncInputStreamManager(int node_id, std::vector<StreamConfig> &input_streams,
                                         std::vector<int> &output_stream_id_list, uint32_t max_queue_size,
                                         InputStreamManagerCallBack &callback);

        std::string type() override;

        NodeReadiness get_node_readiness(int64_t &min_timestamp) override;

        bool fill_task_input(Task &task) override;

    private:
        int64_t zeropoint_offset_;
        int64_t first_frame_offset_;
        int64_t next_timestamp_;

        std::map<int, std::queue<Packet> > cache_;
        std::map<int, Packet> last_pkt_;
        std::set<int> stream_paused_;
    };

    int create_input_stream_manager(
            std::string const &manager_type, int node_id, std::vector<StreamConfig> input_streams,
            std::vector<int> output_stream_id_list, InputStreamManagerCallBack callback,
            uint32_t queue_size_limit, std::shared_ptr<InputStreamManager> &input_stream_manager);
END_BMF_ENGINE_NS
#endif //BMF_INPUT_STREAM_MANAGER_H
