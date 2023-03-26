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
#include "../include/common.h"
#include "../include/graph_config.h"
#include "../include/output_stream_manager.h"

#include <bmf/sdk/packet.h>

#include "gtest/gtest.h"

USE_BMF_ENGINE_NS
USE_BMF_SDK_NS
TEST(output_stream_manager, propagate_packets) {
    int stream_id = 1;
    std::string name = "audio";
    std::vector<StreamConfig> output_stream_names;
    StreamConfig a;
    a.identifier = "audio";
    output_stream_names.push_back(a);
    OutputStreamManager output_stream_manager = OutputStreamManager(output_stream_names);
    std::shared_ptr<SafeQueue<Packet> > packets = std::make_shared<SafeQueue<Packet> >();
    Packet packet(0);
    packet.set_timestamp(10);
    packets->push(packet);
    output_stream_manager.propagate_packets(0, packets);
}