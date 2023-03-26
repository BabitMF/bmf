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

#ifndef BMF_INPUT_STREAM_H
#define BMF_INPUT_STREAM_H

#include <bmf/sdk/packet.h>
#include "graph_config.h"
#include "safe_queue.h"

#include <queue>
#include <condition_variable>

BEGIN_BMF_ENGINE_NS
    USE_BMF_SDK_NS

    class InputStream {
    public:
        InputStream(int stream_id, std::string const &identifier, std::string const &alias, std::string const &notify,
                    int node_id, std::function<void(int, bool)> &throttled_cb, int max_queue_size = 5);

        InputStream(int stream_id, StreamConfig &stream_config, int node_id,
                    std::function<void(int, bool)> &throttled_cb, size_t max_queue_size = 5);

        InputStream(InputStream const &input_stream) = delete;

        InputStream &operator=(InputStream const &sq) = delete;

        int add_packets(std::shared_ptr<SafeQueue<Packet> > &packet);

        Packet pop_packet_at_timestamp(int64_t timestamp);

        Packet pop_next_packet(bool block = true);

        bool is_empty();

        int get_id();

        bool is_full();

        int64_t get_time_bounding();

        void set_connected(bool connected);

        bool is_connected();

        std::string get_identifier();

        std::string get_alias();

        std::string get_notify();

        void clear_queue();

        bool get_min_timestamp(int64_t &min_timestamp);

        bool get_block();

        void set_block(bool block);

        void probe_eof(bool probed);

        void wait_on_empty();

    public:
        int max_queue_size_;
        std::shared_ptr<SafeQueue<Packet> > queue_;
        std::string identifier_;
        std::string notify_;
        std::string alias_;
        int stream_id_;
        int node_id_;
        std::string stream_manager_name_;
        int64_t next_time_bounding_;
        mutable std::mutex mutex_;
        std::condition_variable fill_packet_event_;
        std::mutex stream_m_;
        std::mutex probe_m_;
        std::condition_variable stream_ept_;
        bool block_ = false;
        std::function<void(int, bool)> throttled_cb_;
        bool connected_ = false;
        bool probed_ = false;
    };
END_BMF_ENGINE_NS

#endif //BMF_INPUT_STREAM_H
