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

#include "assemble_module.h"
#include <bmf/sdk/log.h>

AssembleModule::AssembleModule(int node_id, JsonParam json_param)
    : Module(node_id, json_param) {
    BMFLOG_NODE(BMF_INFO, node_id_) << "pass through module";
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

    while (!tem_queue[queue_index_]->empty()) {
        Packet packet;
        auto queue = tem_queue.find(queue_index_);
        
        if (in_eof_[queue_index_] == true)
            continue;
        
        if(task.pop_packet_from_input_queue(queue_index_, packet)) {
            task.fill_output_packet(0, packet);
            if (packet.timestamp() == BMF_EOF) {
                in_eof_[queue_index_] = true;
            }
            BMFLOG_NODE(BMF_INFO, node_id_)
                << "get packet :" << packet.timestamp()
                << " data:" << packet.type_info().name
                << " in queue:" << queue_index_;
        }

        // while (task.pop_packet_from_input_queue(queue_index_, packet)) {
        //     task.fill_output_packet(queue_index_, packet);
        //     if (packet.timestamp() == BMF_EOF) {
        //         in_eof_[queue_index_] = true;
        //     }
        //     BMFLOG_NODE(BMF_INFO, node_id_)
        //         << "get packet :" << packet.timestamp()
        //         << " data:" << packet.type_info().name
        //         << " in queue:" << queue_index_;
        // }
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
