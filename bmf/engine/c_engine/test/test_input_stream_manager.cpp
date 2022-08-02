#include "call_back_for_test.h"

#include "../include/common.h"
#include "../include/graph_config.h"
#include "../include/input_stream_manager.h"

#include "gtest/gtest.h"
//void scheduler_node(Task task){
//    std::cout<<"scheduler node task:"<<task.get_timestamp()<<std::endl;
//}
//bool schedule_node(){
//    std::cout<<"schedule node"<<std::endl;
//}
//int g_manager_node_id = 0;
//bool g_manager_is_add = false;
//void add_or_remove_node_manager(int node_id,bool is_add){
//    g_manager_node_id = node_id;
//    g_manager_is_add = is_add;
//    std::cout<<"add_or_remove_node:"<<node_id<<" is add :"<<is_add<<std::endl;
//}
USE_BMF_ENGINE_NS
USE_BMF_SDK_NS

TEST(input_stream_manager, schedule_node) {
    int node_id = 1;
    std::vector<StreamConfig> input_stream_names;
    StreamConfig v, a;
    v.identifier = "video";
    a.identifier = "audio";
    input_stream_names.push_back(v);
    input_stream_names.push_back(a);
//    std::vector<std::string> output_stream_names;
//    output_stream_names.push_back("encode");
    std::vector<int> output_stream_id_list;
    output_stream_id_list.push_back(0);
    CallBackForTest call_back;
    std::function<void(int, bool)> throttled_cb = call_back.callback_add_or_remove_node_;
    std::function<void(int, bool)> sched_required = call_back.callback_add_or_remove_node_;
    std::function<void(Task &)> scheduler_cb = call_back.callback_scheduler_to_schedule_node_;
    std::function<bool()> notify_cb = call_back.callback_node_to_schedule_node_;
    std::function<bool()> node_is_closed_cb = call_back.callback_node_is_closed_cb_;
    InputStreamManagerCallBack callback;
    callback.scheduler_cb = scheduler_cb;
    callback.notify_cb = notify_cb;
    callback.throttled_cb = throttled_cb;
    callback.sched_required = sched_required;
    callback.node_is_closed_cb = node_is_closed_cb;
    ImmediateInputStreamManager immediate_input_stream_manager(node_id, input_stream_names, output_stream_id_list,
                                                               5, callback);
    int stream_id = 1;
    std::shared_ptr<SafeQueue<Packet> > packets = std::make_shared<SafeQueue<Packet> >();
    Packet packet(0);
    packet.set_timestamp(10);
    packets->push(packet);
    immediate_input_stream_manager.add_packets(stream_id, packets);
    immediate_input_stream_manager.schedule_node();
}
