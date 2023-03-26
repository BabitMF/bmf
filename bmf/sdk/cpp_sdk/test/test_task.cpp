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

#include <gtest/gtest.h>

USE_BMF_SDK_NS
TEST(task, task) {
    std::vector<int> input_labels;
    std::vector<int> output_labels;
    input_labels.push_back(0);
    input_labels.push_back(1);
    output_labels.push_back(0);
    Task task(7, input_labels, output_labels);
    EXPECT_EQ(task.get_node(), 7);
    task.set_timestamp(10);
    EXPECT_EQ(task.timestamp(), 10);

    EXPECT_EQ(task.get_input_stream_ids().size(), 2);
    EXPECT_EQ(task.get_output_stream_ids().size(), 1);

    //TEST input stream
    {
        Packet pkt0(0);
        pkt0.set_timestamp(9);
        task.fill_input_packet(0, pkt0);

        Packet pkt1(0);
        pkt1.set_timestamp(9);
        task.fill_input_packet(1, pkt1);
        
        Packet pkt_output;
        std::cout << task.pop_packet_from_input_queue(0, pkt_output) << std::endl;
        EXPECT_EQ(pkt_output.timestamp(), 9);
        
        PacketQueueMap input_stream_map = task.get_inputs();
        EXPECT_EQ(input_stream_map[0]->size(), 0);
        EXPECT_EQ(input_stream_map[1]->size(), 1);

    }
    //TEST output stream
    {
        Packet pkt(0);
        pkt.set_timestamp(88);
        task.fill_output_packet(0,pkt);
        Packet pkt_output;
        task.pop_packet_from_out_queue(0, pkt_output);
        EXPECT_EQ(pkt_output.timestamp(), 88);
        PacketQueueMap output_stream_map = task.get_outputs();
        EXPECT_EQ(output_stream_map[0]->size(), 0);
    }

}