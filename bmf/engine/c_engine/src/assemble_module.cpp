/*
 * Copyright 2024 Babit Authors
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

#include "../include/assemble_module.h"
#include <bmf/sdk/log.h>

AssembleModule::AssembleModule(int node_id, JsonParam json_param)
    : Module(node_id, json_param) {
    BMFLOG_NODE(BMF_INFO, node_id_) << "assemble module";
    last_input_num_ = 0;
    last_output_num_ = 0;
    queue_index_ = 0;
    return;
}

int AssembleModule::process(Task &task) {
    if (task.get_inputs().size() != last_input_num_) {
        BMFLOG_NODE(BMF_DEBUG, node_id_)
            << "Input Queue size changed from " << last_input_num_ << " to "
            << task.get_inputs().size();
        last_input_num_ = task.get_inputs().size();
        // init queue_map_ 
        for (int i = 0; i < last_input_num_; i++) {
            std::shared_ptr<std::queue<Packet>> tmp_queue = 
                std::make_shared<std::queue<Packet>>();
            queue_map_.insert(
                std::pair<int, std::shared_ptr<std::queue<Packet>>>(
                    i, tmp_queue));
        }
    }
    if (task.get_outputs().size() != last_output_num_) {
        BMFLOG_NODE(BMF_DEBUG, node_id_)
            << "Output Queue size changed from " << last_output_num_ << " to "
            << task.get_outputs().size();
        last_output_num_ = task.get_outputs().size();
    }

    if (in_eof_.size() != task.get_inputs().size()) {
        in_eof_.clear();
        for (auto input_queue : task.get_inputs())
            in_eof_[input_queue.first] = false;
    }
    // assemble data from multi input queue 
    auto tem_queue = task.get_inputs();
    // cache pkts into queue_map_ 
    for (size_t i = 0; i < tem_queue.size(); i++) {
        while (!tem_queue[i]->empty()) {
            auto q = tem_queue[i];
            Packet pkt = q->front();
            q->pop();
            queue_map_[i]->push(pkt);
        }
    }
    
    while (!queue_map_[queue_index_]->empty()) {
        // pass through pkts 
        Packet packet;
        auto queue = queue_map_.find(queue_index_);
        
        if (in_eof_[queue_index_] == true)
            continue;
        
        packet = queue->second->front();
        queue->second->pop();

        task.fill_output_packet(0, packet);
        if (packet.timestamp() == BMF_EOF) {
            in_eof_[queue_index_] = true;
        }
        BMFLOG_NODE(BMF_DEBUG, node_id_)
            << "get packet :" << packet.timestamp()
            << " data:" << packet.type_info().name
            << " in queue:" << queue_index_;

        queue_index_ = (queue_index_ + 1) % queue_map_.size();   
    }

    bool all_eof = true;
    for (auto f_eof : in_eof_) {
        if (f_eof.second == false) {
            all_eof = false;
            break;
        }
    }
    if (all_eof)
        task.set_timestamp(DONE);

    return 0;
}

int AssembleModule::reset() {
    in_eof_.clear();
    return 0;
}

int AssembleModule::close() { return 0; }
