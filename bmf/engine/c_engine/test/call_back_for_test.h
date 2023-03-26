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

#ifndef BMF_CALL_BACK_FOR_TEST_H
#define BMF_CALL_BACK_FOR_TEST_H

#include <bmf/sdk/task.h>

USE_BMF_SDK_NS

class CallBackForTest {
public:
    CallBackForTest();

    void add_or_remove_node(int node_id, bool is_add);

    bool node_to_schedule_node();

    void scheduler_to_schedule_node(Task &task);

    bool node_is_closed_cb();

    int node_id_ = 0;
    bool is_add_ = false;
    int scheduler_to_schedule_node_time_ = 0;
    int node_to_schedule_node_time_ = 0;
    std::function<void(int, bool)> callback_add_or_remove_node_ = NULL;
    std::function<bool()> callback_node_to_schedule_node_ = NULL;
    std::function<void(Task &)> callback_scheduler_to_schedule_node_ = NULL;
    std::function<bool()> callback_node_is_closed_cb_ = NULL;

};


#endif //BMF_CALL_BACK_FOR_TEST_H
