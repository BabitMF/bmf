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

#include "pass_through_module.h"
#include <bmf/sdk/log.h>

PassThroughModule::PassThroughModule(int node_id, JsonParam json_param) : Module(node_id, json_param) {
    BMFLOG_NODE(BMF_INFO, node_id_) << "pass through module";
    last_input_num_ = 0;
    last_output_num_ = 0;
    return;
}

int PassThroughModule::process(Task &task) {
    if (task.get_inputs().size() != last_input_num_) {
        BMFLOG_NODE(BMF_INFO, node_id_) << "Input Queue size changed from " << last_input_num_
            << " to " << task.get_inputs().size();
        last_input_num_ = task.get_inputs().size();
    }
    if (task.get_outputs().size() != last_output_num_) {
        BMFLOG_NODE(BMF_INFO, node_id_) << "Output Queue size changed from " << last_output_num_
            << " to " << task.get_outputs().size();
        last_output_num_ = task.get_outputs().size();
    }

    if (in_eof_.size() != task.get_inputs().size()) {
        in_eof_.clear();
        for (auto input_queue:task.get_inputs())
            in_eof_[input_queue.first] = false;
    }

    for (auto input_queue:task.get_inputs()) {
        Packet packet;

        if (in_eof_[input_queue.first] == true)
            continue;

        while (task.pop_packet_from_input_queue(input_queue.first, packet)) {
            task.fill_output_packet(input_queue.first, packet);
            if (packet.timestamp() == BMF_EOF) {
                in_eof_[input_queue.first] = true;
            }
            BMFLOG_NODE(BMF_INFO, node_id_) << "get packet :" << packet.timestamp() << " data:"
                                            << packet.type_info().name << " in queue:"
                                            << input_queue.first;
        }
    }

    bool all_eof = true;
    for (auto f_eof:in_eof_) {
        if (f_eof.second == false) {
            all_eof = false;
            break;
        }
    }
    if (all_eof)
        task.set_timestamp(DONE);

    return 0;
}

int PassThroughModule::reset() {
    in_eof_.clear();
    return 0;
}

int PassThroughModule::close() {
    return 0;
}
