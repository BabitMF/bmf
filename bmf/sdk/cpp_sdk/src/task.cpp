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

#include <bmf/sdk/task.h>
#include <bmf/sdk/trace.h>

#include <iostream>

BEGIN_BMF_SDK_NS
    Task::Task(int node_id, std::vector<int> input_stream_id_list, std::vector<int> output_stream_id_list) {
        init(node_id, input_stream_id_list, output_stream_id_list);
    }

    Task::Task(const Task &rhs) {
        this->node_id_ = rhs.node_id_;
        this->timestamp_ = rhs.timestamp_;
        this->inputs_queue_ = rhs.inputs_queue_;
        this->outputs_queue_ = rhs.outputs_queue_;
    }

    Task::Task(Task &&rhs) : Task() {
        swap(*this, rhs);
    }

    Task &Task::operator=(Task rhs) {
        swap(*this, rhs);
        return *this;
    }

    BMF_API void swap(Task &target, Task &source) {
        using std::swap;

        std::swap(target.node_id_, source.node_id_);
        std::swap(target.timestamp_, source.timestamp_);
        auto itp = std::move(source.inputs_queue_);
        source.inputs_queue_ = std::move(target.inputs_queue_);
        target.inputs_queue_ = std::move(itp);
        auto otp = std::move(source.outputs_queue_);
        source.outputs_queue_ = std::move(target.outputs_queue_);
        target.outputs_queue_ = std::move(otp);
    }

    void Task::init(int node_id, std::vector<int> input_stream_id_list, std::vector<int> output_stream_id_list) {
        node_id_ = node_id;
        timestamp_ = 0;
        for (int i = 0; i < input_stream_id_list.size(); i++) {
            std::shared_ptr<std::queue<Packet>>
                    tmp_queue = std::make_shared<std::queue<Packet> >();
            inputs_queue_.insert(
                    std::pair<int, std::shared_ptr<std::queue<Packet>>>(
                            input_stream_id_list[i], tmp_queue)
            );
        }
        for (int i = 0; i < output_stream_id_list.size(); i++) {
            std::shared_ptr<std::queue<Packet>>
                    tmp_queue = std::make_shared<std::queue<Packet> >();
            outputs_queue_.insert(
                    std::pair<int, std::shared_ptr<std::queue<Packet>>>(
                            output_stream_id_list[i], tmp_queue)
            );
        }
    }

    bool Task::fill_input_packet(int stream_id, Packet packet) {
        PacketQueueMap::iterator it = inputs_queue_.find(stream_id);
        if (it == inputs_queue_.end()) {
            return false;
        }
        it->second->push(packet);
        return true;
    }

    bool Task::fill_output_packet(int stream_id, Packet packet) {
        PacketQueueMap::iterator it = outputs_queue_.find(stream_id);
        if (it == outputs_queue_.end()) {
            return false;
        }
        it->second->push(packet);
        return true;
    }

    bool Task::pop_packet_from_input_queue(int stream_id, Packet &packet) {
        PacketQueueMap::iterator it = inputs_queue_.find(stream_id);
        if (it == inputs_queue_.end()) {
            return false;
        }
        std::shared_ptr<std::queue<Packet> > q = it->second;
        if (q->empty()) {
            return false;
        }
        packet = q->front();
        q->pop();
        BMF_TRACE_THROUGHPUT(stream_id, node_id_, q->size());
        return true;
    }

    bool Task::pop_packet_from_out_queue(int stream_id, Packet &packet) {
        auto it = outputs_queue_.find(stream_id);
        if (it == outputs_queue_.end()) {
            return false;
        }
        std::shared_ptr<std::queue<Packet> > q = it->second;
        if (q->empty()) {
            return false;
        }
        packet = q->front();
        q->pop();
        return true;
    }

    bool Task::input_queue_empty(int stream_id)
    {
        auto it = inputs_queue_.find(stream_id);
        if (it == inputs_queue_.end()) {
            return true;
        }
        return it->second->empty();
    }

    bool Task::output_queue_empty(int stream_id)
    {
        auto it = outputs_queue_.find(stream_id);
        if (it == outputs_queue_.end()) {
            return true;
        }
        return it->second->empty();
    }

    int64_t Task::timestamp() const {
        return timestamp_;
    }

    void Task::set_timestamp(int64_t t) {
        timestamp_ = t;
    }

    std::vector<int> Task::get_input_stream_ids() {
        std::vector<int> input_stream_ids;
        for (auto input :inputs_queue_) {
            input_stream_ids.push_back(input.first);
        }
        return input_stream_ids;
    }

    std::vector<int> Task::get_output_stream_ids() {
        std::vector<int> output_stream_ids;
        for (auto output :outputs_queue_) {
            output_stream_ids.push_back(output.first);
        }
        return output_stream_ids;
    }

    PacketQueueMap &Task::get_inputs() {
        return inputs_queue_;
    }

    PacketQueueMap &Task::get_outputs() {
        return outputs_queue_;
    }

    int Task::get_node() {
        return node_id_;
    }

END_BMF_SDK_NS
