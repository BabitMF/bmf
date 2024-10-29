/*
 * Copyright 2024 Babit Authors
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

#include "../include/split_module.h"
#include <bmf/sdk/log.h>

SplitModule::SplitModule(int node_id, JsonParam json_param)
    : Module(node_id, json_param) {
    BMFLOG_NODE(BMF_INFO, node_id_) << "split module";
    last_input_num_ = 0;
    last_output_num_ = 0;
    stream_index_ = 0;
    queue_index_ = 0;
    return;
}

int SplitModule::process(Task &task) {
    if (task.get_inputs().size() != last_input_num_) {
        BMFLOG_NODE(BMF_INFO, node_id_)
            << "Input Queue size changed from " << last_input_num_ << " to "
            << task.get_inputs().size();
        last_input_num_ = task.get_inputs().size();
        if (last_input_num_ > 1) {
            BMFLOG_NODE(BMF_ERROR, node_id_)
                << "Input Queue size > 1 may be meet error!";
            close();
        }
    }
    if (task.get_outputs().size() != last_output_num_) {
        BMFLOG_NODE(BMF_INFO, node_id_)
            << "Output Queue size changed from " << last_output_num_ << " to "
            << task.get_outputs().size();
        last_output_num_ = task.get_outputs().size();
        // init queue_map_ 
        for (int i = 0; i < last_output_num_; i++) {
            std::shared_ptr<bmf_engine::SafeQueue<Packet>> tmp_queue = 
                std::make_shared<bmf_engine::SafeQueue<Packet>>();
            queue_map_.insert(
                std::pair<int, std::shared_ptr<bmf_engine::SafeQueue<Packet>>>(
                    i, tmp_queue));
        }
    }

    if (in_eof_.size() != task.get_inputs().size()) {
        in_eof_.clear();
        for (auto input_queue : task.get_inputs())
            in_eof_[input_queue.first] = false;
    }

    // Data cache into queue_map
    auto tem_queue = task.get_inputs();
    Packet pkt;
    while (!tem_queue[0]->empty()) {
        Packet pkt = tem_queue[0]->front();
        tem_queue[0]->pop();
        queue_map_[stream_index_]->push(pkt);
        if (pkt.timestamp() == EOS or pkt.timestamp() == BMF_EOF) {
            /* add EOF pkt for multi downstream node */
            for (size_t i = 1; i < task.get_outputs().size(); i++) {
                queue_map_[i]->push(Packet::generate_eof_packet());
            }
        }
        stream_index_ = (stream_index_ + 1) % task.get_outputs().size();
    }

    // Data Splitting
    while (!queue_map_[queue_index_]->empty()) {
        
        if (in_eof_[queue_index_] == true)
            continue;
        
        auto queue = queue_map_.find(queue_index_);
        if (queue->second->pop(pkt)) {
            // fill splitted pkt into multi output stream
            task.fill_output_packet(queue_index_, pkt);
            if (pkt.timestamp() == BMF_EOF) {
                in_eof_[queue_index_] = true;
            }
            BMFLOG_NODE(BMF_DEBUG, node_id_)
                << "get packet :" << pkt.timestamp()
                << " data:" << pkt.type_info().name
                << " in queue:" << queue_index_;

            queue_index_ = (queue_index_ + 1) % task.get_outputs().size();
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

int SplitModule::reset() {
    in_eof_.clear();
    return 0;
}

int SplitModule::close() { return 0; }
