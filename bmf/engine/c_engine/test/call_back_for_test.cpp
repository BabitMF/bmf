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

CallBackForTest::CallBackForTest() {
    callback_add_or_remove_node_ = std::bind(&CallBackForTest::add_or_remove_node, this, std::placeholders::_1,
                                             std::placeholders::_2);
    callback_node_to_schedule_node_ = std::bind(&CallBackForTest::node_to_schedule_node, this);
    callback_scheduler_to_schedule_node_ = std::bind(&CallBackForTest::scheduler_to_schedule_node, this,
                                                     std::placeholders::_1);
    callback_node_is_closed_cb_ = std::bind(&CallBackForTest::node_is_closed_cb, this);
}

void CallBackForTest::add_or_remove_node(int node_id, bool is_add) {
    node_id_ = node_id;
    is_add_ = is_add;
}

void CallBackForTest::scheduler_to_schedule_node(Task &task) {
    scheduler_to_schedule_node_time_++;
}

bool CallBackForTest::node_to_schedule_node() {
    node_to_schedule_node_time_++;
    return true;
}

bool CallBackForTest::node_is_closed_cb() {
    return false;
}