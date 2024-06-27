/*
 * Copyright 2023 Babit Authors
 *
 * This file is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This file is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
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
        BMFLOG_NODE(BMF_INFO, node_id_)
            << "Input Queue size changed from " << last_input_num_ << " to "
            << task.get_inputs().size();
        last_input_num_ = task.get_inputs().size();
        /* init queue_map_ */
        for (int i = 0; i < last_input_num_; i++) {
            std::shared_ptr<bmf_engine::SafeQueue<Packet>> tmp_queue = 
                std::make_shared<bmf_engine::SafeQueue<Packet>>();
            queue_map_.insert(
                std::pair<int, std::shared_ptr<bmf_engine::SafeQueue<Packet>>>(
                    i, tmp_queue));
        }
    }
    if (task.get_outputs().size() != last_output_num_) {
        BMFLOG_NODE(BMF_INFO, node_id_)
            << "Output Queue size changed from " << last_output_num_ << " to "
            << task.get_outputs().size();
        last_output_num_ = task.get_outputs().size();
    }

    if (in_eof_.size() != task.get_inputs().size()) {
        in_eof_.clear();
        for (auto input_queue : task.get_inputs())
            in_eof_[input_queue.first] = false;
    }
    /* assemble data from multi input queue */
    auto tem_queue = task.get_inputs();
    /* cache pkts into queue_map_ */
    for (size_t i = 0; i < tem_queue.size(); i++) {
        while (!tem_queue[i]->empty()) {
            auto q = tem_queue[i];
            Packet pkt = q->front();
            q->pop();
            queue_map_[i]->push(pkt);
        }
    }
    
    while (!queue_map_[queue_index_]->empty()) {
        /* pass through pkts */
        Packet packet;
        auto queue = queue_map_.find(queue_index_);
        
        if (in_eof_[queue_index_] == true)
            continue;
        
        // if(task.pop_packet_from_input_queue(queue_index_, packet)) {
        if (queue->second->pop(packet)) {
            task.fill_output_packet(0, packet);
            if (packet.timestamp() == BMF_EOF) {
                in_eof_[queue_index_] = true;
            }
            BMFLOG_NODE(BMF_INFO, node_id_)
                << "get packet :" << packet.timestamp()
                << " data:" << packet.type_info().name
                << " in queue:" << queue_index_;

            queue_index_ = (queue_index_ + 1) % queue_map_.size();
        }
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
