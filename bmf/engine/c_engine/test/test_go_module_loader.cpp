#include <filesystem>
#include "gtest/gtest.h"
#include "module_factory.h"
#include <bmf/sdk/video_frame.h>


#ifndef BMF_ENABLE_MOBILE

using namespace bmf_engine;
using namespace bmf_sdk;

TEST(go_module, module_loader)
{
    auto module_path = "../lib/go_pass_through.so";
    if(!std::filesystem::exists(module_path)){
        GTEST_SKIP();
    }

    std::shared_ptr<Module> module;  
    JsonParam option;
    ModuleFactory::create_module(
        "PassThrough", 42, option, "go", module_path,
        "", module);

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

    //process
    std::vector<int> iids{1, 2}, oids{2,3};
    Task task(42, iids, oids);

    //fill inputs
    for(int i = 0; i < 10; ++i){
        auto vf0 = VideoFrame::make(1920, 1080, 3);
        auto vf1 = VideoFrame::make(1280, 720, 3);
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

    for(int i = 0; i < 10; ++i){
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

#endif //BMF_ENABLE_MOBILE