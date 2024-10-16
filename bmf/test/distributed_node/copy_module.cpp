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
#include "copy_module.h"

int CopyModule::process(Task &task) {
    //usleep(30*MS);
    PacketQueueMap &input_queue_map = task.get_inputs();
    PacketQueueMap::iterator it;

    // process all input queues
    for (it = input_queue_map.begin(); it != input_queue_map.end(); it++) {
        // input stream label
        int label = it->first;

        // input packet queue
        Packet pkt;
        // process all packets in one input queue
        while (task.pop_packet_from_input_queue(label, pkt)) {
            // Get a input packet

            // if packet is eof, set module done
            if (pkt.timestamp() == BMF_EOF) {
                task.set_timestamp(DONE);
                task.fill_output_packet(label, Packet::generate_eof_packet());
                return 0;
            }

            auto output_pkt = copy(pkt);
            // copy timestamp
            output_pkt.set_timestamp(pkt.timestamp());

            task.fill_output_packet(label, output_pkt);

            // std::cout << "node: copy module " << node_id_ << "pkt's timestamp: " << output_pkt.timestamp() << std::endl;

        }

    }
    return 0;
}

Packet CopyModule::copy(Packet &pkt){
    // Get packet data
    // Here we should know the data type in packet
    auto vframe = pkt.get<VideoFrame>();

    // Deep copy
    VideoFrame vframe_out = VideoFrame(vframe.frame().clone());
    vframe_out.copy_props(vframe);

    // Add output frame to output queue
    return Packet(vframe_out);
}

REGISTER_MODULE_CLASS(CopyModule)
