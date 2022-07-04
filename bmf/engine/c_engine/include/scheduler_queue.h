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

#ifndef BMF_SCHEDULER_QUEUE_H
#define BMF_SCHEDULER_QUEUE_H

#include <thread>
#include <atomic>
#include <condition_variable>
#include "node.h"
#include <bmf/sdk/task.h>
#include <exception>
#include <stdexcept>

BEGIN_BMF_ENGINE_NS
    USE_BMF_SDK_NS
    enum class State {
        INITED = 1,
        RUNNING = 2,
        TERMINATING = 3,
        TERMINATED = 4,
        PAUSED = 5,
    };

    class Item {
    public:
        int priority;
        Task task;

//        Item(int priority = 0, Task task = Task()) {
//            this->priority = priority;
//            this->task = task;
//        }

        Item() {
            this->priority = 0;
            this->task = Task();
        }

        Item(Item const &rhs) {
            this->priority = rhs.priority;
            this->task = rhs.task;
        }

        Item(Item &&rhs) noexcept {
            this->priority = rhs.priority;
            this->task = std::move(rhs.task);
        }

        Item(int priority, Task &&task) {
            this->priority = priority;
            this->task = std::move(task);
        }

        Item(int priority, Task const &task) {
            this->priority = priority;
            this->task = task;
        }

        friend void swap(Item &target, Item &source) {
            using std::swap;

            swap(target.priority, source.priority);
            swap(target.task, source.task);
        }

        Item &operator=(Item rhs) {
            swap(*this, rhs);
            return *this;
        }
    };

    class SchedulerQueueCallBack {
    public:
        std::function<int(int, std::shared_ptr<Node> &)> get_node_;
    };

    class SchedulerQueue {
    public:
        bool paused_ = false;
        int id_;
        State paused_state_;
        std::thread exec_thread_;
        State state_;
        bool exception_catch_flag_ = false;
        std::exception_ptr eptr_ ;
        int64_t start_time_; //TODO: unused
        int64_t wait_duration_ = 0;
        int64_t wait_cnt_ = 0;
        SchedulerQueueCallBack callback_;
        // TODO:
        SafePriorityQueue<Item> queue_;
//    SafeQueue<Item> queue_;
        std::condition_variable con_var_;
        std::mutex con_var_mutex_;

        SchedulerQueue(int id, SchedulerQueueCallBack callback);

        Item pop_task();

        int add_task(Task &task, int priority);

        int remove_node_task(int node_id);

        int exec_loop();

        int exec(Task &task);

        void internal_pause();

        void pause();

        void resume();

        int start();

        int close();
    };
END_BMF_ENGINE_NS

#endif //BMF_SCHEDULER_QUEUE_H
