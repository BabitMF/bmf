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

#include "../include/common.h"
#include "../include/node.h"

#include "gtest/gtest.h"

USE_BMF_ENGINE_NS
USE_BMF_SDK_NS

int init_node(std::shared_ptr<Node> &node) {
    int node_id = 1;
    JsonParam json_param;
    json_param.load("../files/node.json");
    NodeConfig node_config = NodeConfig(json_param);
    CallBackForTest callback_for_test;
    NodeCallBack callback;
    callback.scheduler_cb = callback_for_test.callback_scheduler_to_schedule_node_;
    callback.throttled_cb = callback_for_test.callback_add_or_remove_node_;
    callback.sched_required = callback_for_test.callback_add_or_remove_node_;
    std::shared_ptr<Module> pre_allocated_modules = NULL;
    BmfMode mode = BmfMode::NORMAL_MODE;
    node = std::make_shared<Node>(node_id, node_config, callback, pre_allocated_modules, mode, nullptr);
    return 0;
}

TEST(node, schedule_node) {
    std::shared_ptr<Node> node;
    init_node(node);
//    node->schedule_node();
}

TEST(node, process_node) {
    std::shared_ptr<Node> node;
    init_node(node);
    Task task = Task();
    node->process_node(task);
}
