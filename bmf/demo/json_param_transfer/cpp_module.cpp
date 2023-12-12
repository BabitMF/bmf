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
#include "cpp_module.h"

int cpp_module::process(Task &task) {
    PacketQueueMap &input_queue_map = task.get_inputs();
    PacketQueueMap &output_queue_map = task.get_outputs();
    PacketQueueMap::iterator it;

    for (it = input_queue_map.begin(); it != input_queue_map.end(); it++) {
        // input stream label
        int label = it->first;
        Packet pkt;
        // process all packets in one input queue
        while (task.pop_packet_from_input_queue(label, pkt)) {
            // if packet is eof, set module done
            if (pkt.timestamp() == BMF_EOF) {
                task.set_timestamp(DONE);
                auto out_it = output_queue_map.begin();
                if (out_it != output_queue_map.end()) {
                    task.fill_output_packet(out_it->first, Packet::generate_eof_packet());
                }
                return 0;
            }

            // parse the input json param if there's JsonParam
            if (pkt.is<JsonParam>()) {
                auto jp = pkt.get<JsonParam>();
                std::string jstring = jp.dump();
                BMFLOG(BMF_INFO) << "Cpp module parsed json: " << jstring;     
            }

            JsonParam transfer_json;
            transfer_json.parse(fmt::format("{{\"input_id\": {}, \"frame_number\": {}}}",
                                           label, frame_number++));
            auto output_pkt = Packet(transfer_json);
            auto out_it = output_queue_map.begin();
            if (out_it != output_queue_map.end()) {
                task.fill_output_packet(out_it->first, output_pkt);
            }
        }
    }
    return 0;
}
REGISTER_MODULE_CLASS(cpp_module)
