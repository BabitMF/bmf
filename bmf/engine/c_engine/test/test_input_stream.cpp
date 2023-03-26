#include "call_back_for_test.h"

#include "../include/common.h"
#include "../include/input_stream.h"

#include "gtest/gtest.h"

USE_BMF_ENGINE_NS
USE_BMF_SDK_NS
TEST(input_stream, add_normal_packets) {

    int stream_id = 1;
    std::string name = "video";
    int max_queue_size = 5;
    int node_id = 1;
    CallBackForTest call_back;
    std::function<void(int, bool)> throttled_cb = call_back.callback_add_or_remove_node_;
    InputStream input_stream = InputStream(stream_id, name, "", "", node_id, throttled_cb, max_queue_size);
//    SafeQueue<Packet> packets;
    std::shared_ptr<SafeQueue<Packet> > packets = std::make_shared<SafeQueue<Packet> >();
    Packet pkt(0);
    pkt.set_timestamp(10);
    packets->push(pkt);
    input_stream.add_packets(packets);
    EXPECT_EQ(input_stream.get_time_bounding(), 11);
    EXPECT_EQ(call_back.node_id_, 0);
    EXPECT_EQ(call_back.is_add_, 0);
}

TEST(input_stream, add_eof_packets) {

    int stream_id = 1;
    std::string name = "video";
    int max_queue_size = 5;
    int node_id = 1;
    CallBackForTest call_back;
    std::function<void(int, bool)> throttled_cb = call_back.callback_add_or_remove_node_;
    InputStream input_stream = InputStream(stream_id, name, "", "", node_id, throttled_cb, max_queue_size);
    std::shared_ptr<SafeQueue<Packet> > packets = std::make_shared<SafeQueue<Packet> >();
    Packet pkt0(0);
    pkt0.set_timestamp(10);
    packets->push(pkt0);
    Packet pkt1(0);
    pkt1.set_timestamp(BMF_EOF);
    packets->push(pkt1);
    input_stream.add_packets(packets);
    EXPECT_EQ(input_stream.get_time_bounding(), DONE);
    EXPECT_EQ(call_back.node_id_, 1);
    EXPECT_EQ(call_back.is_add_, 1);
}

TEST(input_stream, pop_packet_at_timestamp) {

}

TEST(input_stream, pop_next_packet) {
    int stream_id = 1;
    std::string name = "video";
    int max_queue_size = 5;
    int node_id = 1;
    CallBackForTest call_back;
    std::function<void(int, bool)> throttled_cb = call_back.callback_add_or_remove_node_;
    InputStream input_stream = InputStream(stream_id, name, "", "", node_id, throttled_cb, max_queue_size);
    std::shared_ptr<SafeQueue<Packet> > packets = std::make_shared<SafeQueue<Packet> >();
    Packet pkt0(0);
    pkt0.set_timestamp(10);
    packets->push(pkt0);
    Packet pkt1(0);
    pkt1.set_timestamp(BMF_EOF);
    packets->push(pkt1);
    input_stream.add_packets(packets);
    Packet pop_packet = input_stream.pop_next_packet(false);
    EXPECT_EQ(pop_packet.timestamp(), 10);
    pop_packet = input_stream.pop_next_packet(false);
    EXPECT_EQ(pop_packet.timestamp(), BMF_EOF);
    EXPECT_EQ(call_back.node_id_, 1);
    EXPECT_EQ(call_back.is_add_, 0);

}