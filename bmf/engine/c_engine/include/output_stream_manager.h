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

#ifndef BMF_OUTPUT_STREAM_MANAGER_H
#define BMF_OUTPUT_STREAM_MANAGER_H

#include "../include/graph_config.h"
#include "../include/output_stream.h"

#include <string>

BEGIN_BMF_ENGINE_NS
    USE_BMF_SDK_NS

    class OutputStreamManager {
    public:
        OutputStreamManager(std::vector<StreamConfig> output_streams);

        bool get_stream(int stream_id, std::shared_ptr<OutputStream> &output_stream);

        int add_stream(std::string name);

        std::vector<int> get_stream_id_list();

        int post_process(Task &task);

        int propagate_packets(int stream_id, std::shared_ptr<SafeQueue<Packet> > packets);

        bool any_of_downstream_full();

        void probe_eof();

        void remove_stream(int stream_id, int mirror_id);

        void wait_on_stream_empty(int stream_id);

        int get_outlink_nodes_id(std::vector<int> &nodes_id);

        std::map<int, std::shared_ptr<OutputStream>> output_streams_;
        std::vector<int> stream_id_list_;

    private:
        int max_id_;
    };

END_BMF_ENGINE_NS
#endif //BMF_OUTPUT_STREAM_MANAGER_H
