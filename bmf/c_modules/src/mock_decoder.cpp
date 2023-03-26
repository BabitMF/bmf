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

#include <unistd.h>
#include "mock_decoder.h"
#include <bmf/sdk/log.h>

MockDecoder::MockDecoder(int node_id, JsonParam json_param) : Module(node_id, json_param) {
    BMFLOG_NODE(BMF_INFO, node_id_) << "init";
    return;
}

int MockDecoder::process(Task &task) {
    BMFLOG_NODE(BMF_INFO, node_id_) << "process";
    number_++;
    for (auto output_queue:task.get_outputs()) {
        std::string data = "hello world";
        auto packet = Packet(data);
        packet.set_timestamp(number_);
        BMFLOG_NODE(BMF_INFO, node_id_) << packet.timestamp() << "data type:" << packet.type_info().name;
        task.fill_output_packet(output_queue.first,packet);

        sleep(1);
        if (number_ == 10) {
            task.fill_output_packet(output_queue.first, Packet::generate_eof_packet());
            task.set_timestamp(DONE);
        }
    }
    BMFLOG_NODE(BMF_INFO, node_id_) << "MockDecoder process result output queue size: " << task.get_outputs()[0]->size()
                                    << std::endl;
    return 0;
}

int MockDecoder::reset() {
    return 0;
}

int MockDecoder::close() {
    return 0;
}
