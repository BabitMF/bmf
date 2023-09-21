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
#include <filesystem>
#include "gtest/gtest.h"
#include "module_factory.h"
#include <bmf/sdk/video_frame.h>

#ifndef BMF_ENABLE_MOBILE

using namespace bmf_engine;
using namespace bmf_sdk;

TEST(go_module, module_loader) {
    std::string module_name = "go_copy_module";

    int node_id = 42;
    JsonParam option;
    std::string module_type;
    std::string module_path;
    std::string module_entry;
    std::shared_ptr<Module> module;
    ModuleFactory::create_module(module_name, node_id, option, module_type,
                                 module_path, module_entry, module);
    EXPECT_EQ(module == nullptr, 0);

    // get module info
    JsonParam module_info;
    module->get_module_info(module_info);
    EXPECT_EQ(module_info.get<std::string>("NodeId"), "42");

    //
    EXPECT_NO_THROW(module->init());
    EXPECT_TRUE(module->need_hungry_check(0));
    EXPECT_TRUE(module->is_hungry(0));
    EXPECT_TRUE(module->is_infinity());

    //
    EXPECT_THROW(module->reset(), std::runtime_error);

    // process
    std::vector<int> iids{1, 2}, oids{2, 3};
    Task task(42, iids, oids);

    // fill inputs
    for (int i = 0; i < 10; ++i) {
        auto rgb = hmp::PixelInfo(hmp::PF_RGB24);
        auto vf0 = VideoFrame::make(1920, 1080, rgb);
        auto vf1 = VideoFrame::make(1280, 720, rgb);
        auto pkt0 = Packet(vf0);
        pkt0.set_timestamp(i);
        auto pkt1 = Packet(vf1);
        pkt1.set_timestamp(i);

        task.fill_input_packet(1, pkt0);
        task.fill_input_packet(2, pkt1);
    }
    task.fill_input_packet(1, Packet::generate_eof_packet());
    task.fill_input_packet(2, Packet::generate_eof_packet());

    EXPECT_NO_THROW(module->process(task));

    EXPECT_EQ(task.timestamp(), Timestamp::DONE);

    for (int i = 0; i < 10; ++i) {
        Packet pkt;
        EXPECT_TRUE(task.pop_packet_from_out_queue(2, pkt));
        auto vf0 = pkt.get<VideoFrame>();
        EXPECT_EQ(vf0.width(), 1920);
        EXPECT_EQ(vf0.height(), 1080);

        EXPECT_TRUE(task.pop_packet_from_out_queue(3, pkt));
        auto vf1 = pkt.get<VideoFrame>();
        EXPECT_EQ(vf1.width(), 1280);
        EXPECT_EQ(vf1.height(), 720);
    }
}

#endif // BMF_ENABLE_MOBILE
