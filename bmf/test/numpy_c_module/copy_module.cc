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
#include "copy_module.h"

using namespace boost::python;
using namespace boost::python::numpy;
void CopyModule::process(Task &task) {
    PacketQueueMap &input_queue_map = task.get_inputs();
    PacketQueueMap::iterator it;

    // process all input queues
    for (it = input_queue_map.begin(); it != input_queue_map.end(); it++) {
        // input stream label
        std::string label = it->first;

        // input packet queue
        std::queue<Packet> in_queue = it->second;

        // process all packets in one input queue
        while (!in_queue.empty()) {
            // Get a input packet
            Packet pkt = in_queue.front();
            in_queue.pop();

            // if packet is eof, set module done
            if (pkt.get_timestamp() == BMF_EOF) {
                task.set_timestamp(DONE);
                task.add_packet_to_out_queue(label, generate_eof_packet());
                return;
            }

            // Get packet data
            // Here we should know the data type in packet

            //            const VideoFrame &frame = pkt.get_data<VideoFrame>();
            //            const ndarray &frame = extract<ndarray>(pkt.data_);
            const ndarray &frame = pkt.get_data<ndarray>();
            // Create output frame
            ndarray new_frame = frame.copy();

            // Add output frame to output queue
            Packet output_pkt;
            output_pkt.set_data<ndarray>(new_frame);
            task.add_packet_to_out_queue(label, output_pkt);
        }
    }
}

BMF_MODULE(copy_module, CopyModule)
