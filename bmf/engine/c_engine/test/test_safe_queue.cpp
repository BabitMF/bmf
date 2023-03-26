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
#include "common.h"

#include "../include/safe_queue.h"

#include "gtest/gtest.h"

USE_BMF_ENGINE_NS
USE_BMF_SDK_NS
TEST(safe_queue, single_thread) {
    SafeQueue<int> safe_queue;
    EXPECT_EQ(safe_queue.size(), 0);
    EXPECT_EQ(safe_queue.empty(), 1);
    safe_queue.push(1);
    int item;
    safe_queue.pop(item);
    EXPECT_EQ(item, 1);

}