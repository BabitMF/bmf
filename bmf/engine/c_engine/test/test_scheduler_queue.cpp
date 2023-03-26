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
#include "../include/scheduler_queue.h"

#include "gtest/gtest.h"
//bool operator<(const Item &lhs, const Item &rhs) {
//    if (lhs.task.get_timestamp()>rhs.task.get_timestamp())
//    {
//        return true;
//    }
//    else if(lhs.task.get_timestamp()==rhs.task.get_timestamp())
//    {
//        if (lhs.task.node_id_>rhs.task.node_id_)
//            return true;
//    }
//    else {
//        return false;
//    }
//}
USE_BMF_ENGINE_NS
USE_BMF_SDK_NS
TEST(scheduler_queue, start) {
    SchedulerQueueCallBack callback;
    std::shared_ptr<SchedulerQueue> scheduler_queue = std::make_shared<SchedulerQueue>(0, callback);
//    scheduler_queue->start();
}
//TEST(scheduler_queue,item){
//    SafePriorityQueue<Item> queue;
//    for (int i=0;i<100;i++){
//        Task task = Task(0,1,1);
//        task.set_timestamp(i);
//        Item item(0,task);
//        queue.push(item);
//    }
//    Item item;
//    while (queue.pop(item)){
//        std::cout<<item.task.get_timestamp()<<std::endl;
//    }
//}