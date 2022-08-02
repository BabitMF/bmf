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
#include "call_back_for_test.h"

#include "../include/output_stream.h"

#include "gtest/gtest.h"

#include <fstream>

USE_BMF_ENGINE_NS
USE_BMF_SDK_NS

std::shared_ptr<CallBackForTest> initInputStreamManager(std::shared_ptr<InputStreamManager> &input_stream_manager) {
    int node_id = 1;
    std::vector<StreamConfig> input_stream_names;
    StreamConfig v, a;
    v.identifier = "video";
    a.identifier = "audio";
    input_stream_names.push_back(v);
    input_stream_names.push_back(a);
    std::vector<int> output_stream_id_list;
    output_stream_id_list.push_back(0);
    auto call_back = std::make_shared<CallBackForTest>();
    std::function<void(int, bool)> throttled_cb = call_back->callback_add_or_remove_node_;
    std::function<void(Task &)> scheduler_cb = call_back->callback_scheduler_to_schedule_node_;
    std::function<void(int, bool)> sched_required = call_back->callback_add_or_remove_node_;
    std::function<bool()> notify_cb = call_back->callback_node_to_schedule_node_;
    std::function<bool()> node_is_closed_cb = call_back->callback_node_is_closed_cb_;
    InputStreamManagerCallBack callback;
    callback.scheduler_cb = scheduler_cb;
    callback.notify_cb = notify_cb;
    callback.throttled_cb = throttled_cb;
    callback.sched_required = sched_required;
    callback.node_is_closed_cb = node_is_closed_cb;
    input_stream_manager = std::make_shared<ImmediateInputStreamManager>(node_id, input_stream_names,
                                                                         output_stream_id_list, 5, callback);
    return call_back;
}

TEST(output_stream, add_mirror_stream) {
    int stream_id = 1;
    std::string name = "audio";
    OutputStream output_stream(stream_id, name);

    std::shared_ptr<InputStreamManager> input_stream_manager;
    auto cb_p = initInputStreamManager(input_stream_manager);
    output_stream.add_mirror_stream(input_stream_manager, stream_id);
}

TEST(output_stream, propagate_packets) {
//    google::InitGoogleLogging("main");
//    google::SetStderrLogging(google::INFO);
//    google::InstallFailureSignalHandler();
//    google::InstallFailureWriter(&SignalHandle);
    int stream_id = 1;
    std::string name = "audio";
    OutputStream output_stream(stream_id, name);

    std::shared_ptr<InputStreamManager> input_stream_manager;
    auto cb_p = initInputStreamManager(input_stream_manager);
    output_stream.add_mirror_stream(input_stream_manager, stream_id);
    std::shared_ptr<SafeQueue<Packet> > packets = std::make_shared<SafeQueue<Packet> >();
    Packet packet(0);
    packet.set_timestamp(10);
    packets->push(packet);
    output_stream.propagate_packets(packets);
}
