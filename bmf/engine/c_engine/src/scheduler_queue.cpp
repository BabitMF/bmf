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
#include "../include/scheduler_queue.h"
#include "../include/node.h"

#include <bmf/sdk/task.h>
#include <bmf/sdk/trace.h>
#include <bmf/sdk/log.h>

BEGIN_BMF_ENGINE_NS
    USE_BMF_SDK_NS


// TODO: Unused Item.priority
    bool operator<(const Item &lhs, const Item &rhs) {
        if (lhs.task.timestamp() > rhs.task.timestamp()) {
            return true;
        } else if (lhs.task.timestamp() == rhs.task.timestamp()) {
            if (lhs.task.node_id_ > rhs.task.node_id_)
                return true;
        } else {
            return false;
        }
    }

    SchedulerQueue::SchedulerQueue(int id, SchedulerQueueCallBack callback)
            : id_(id), callback_(callback), start_time_(0), state_(State::INITED) {}

    Item SchedulerQueue::pop_task() {
        Item item;
        queue_.pop(item);
        return item;
    }

    int SchedulerQueue::add_task(Task &task, int priority) {
        if (state_ == State::TERMINATED)
            return false;
        if (task.timestamp_ != UNSET) {
            std::lock_guard<std::mutex> guard(con_var_mutex_);
            Item item = Item(priority, task);
            queue_.push(item);
            con_var_.notify_one();
            return true;
        }
        return false;
    }

    int SchedulerQueue::remove_node_task(int node_id){
        std::lock_guard<std::mutex> guard(con_var_mutex_);
        SafePriorityQueue<Item> temp_queue;
        while (!queue_.empty()){
            Item item;
            queue_.pop(item);
            if (item.task.node_id_ != node_id){
                temp_queue.push(item);
            }
        }
        while (!temp_queue.empty()){
            Item item;
            temp_queue.pop(item);
            queue_.push(item);
        }
        return 0;
    }

    int SchedulerQueue::exec_loop() {
        while (true) {
            if (paused_)
                internal_pause();
            if (state_ == State::TERMINATING and queue_.empty() || exception_catch_flag_) {
                break;
            }
            {
                std::unique_lock<std::mutex> lk(con_var_mutex_);
                if (queue_.empty()) {
                    wait_cnt_++;
                    int64_t startts;
                    BMF_TRACE(SCHEDULE, ("THREAD_" + std::to_string(id_) + "_WAIT" + "_" + std::to_string(wait_cnt_)).c_str(), START);
                    startts = clock();
                    con_var_.wait(lk, [this] { return this->state_ == State::TERMINATING || !this->queue_.empty(); });
                    wait_duration_ += (clock() - startts);
                    BMF_TRACE(SCHEDULE, ("THREAD_" + std::to_string(id_) + "_WAIT"+ "_" + std::to_string(wait_cnt_)).c_str(), END);
                }
            }
            Item item;
            //printf("DEBUG, schedule queue %d size %d\n", id_, queue_.size());
            while (queue_.pop(item)) {
                try{
                    exec(item.task);
                }catch(...){
                    std::mutex exception_mutex_;
                    const std::lock_guard<std::mutex> lock(exception_mutex_);
                    exception_catch_flag_ = true;
                    this->eptr_ = std::current_exception();
                    break;
                }
                if (paused_)
                    internal_pause();
            }
        }
        return 0;
    }

    void SchedulerQueue::internal_pause() {
        paused_state_ = state_;
        state_ = State::PAUSED;
        while (paused_)
            usleep(1000);
    }

    void SchedulerQueue::pause() {
        paused_ = true;
        if (state_ == State::RUNNING)
            while (state_ != State::PAUSED)
                usleep(1000);
    }

    void SchedulerQueue::resume() {
        if (state_ == State::PAUSED)
            state_ = paused_state_;
        paused_ = false;
    }

    int SchedulerQueue::exec(Task &task) {
        std::shared_ptr<Node> node;
        callback_.get_node_(task.node_id_, node);
//        std::cout << "Working on " << task.node_id_ << std::endl;
//        for (const auto &it:task.get_inputs())
//            std::cout << "input queue id=" << it.first << "  task packet count " << it.second->size() << std::endl;
        auto st = TIMENOW();
        node->process_node(task);
        node->dur += DURATION(TIMENOW() - st);
//        std::cout << "Work done for" << task.node_id_ << std::endl;
//        for (const auto &it:task.get_outputs())
//            std::cout << "output queue id=" << it.first << "  task packet count " << it.second->size() << std::endl;
        return 0;
    }

    int SchedulerQueue::start() {
        state_ = State::RUNNING;
        exec_thread_ = std::thread(&SchedulerQueue::exec_loop, this);
        auto handle = exec_thread_.native_handle();
        std::string thread_name = "schedule_queue"+std::to_string(id_);
#if __APPLE__
        pthread_setname_np(thread_name.c_str());
#else
        pthread_setname_np(handle,thread_name.c_str());
#endif
        return 0;
    }

    int SchedulerQueue::close() {
        if (exec_thread_.joinable()){
            state_ = State::TERMINATING;
            //con_var_ should notify ,otherwise it will block in con_var_.wait()
            con_var_.notify_one();
            exec_thread_.join();
            state_ = State::TERMINATED;
        }
        double duration = (double) wait_duration_ / CLOCKS_PER_SEC;
        //BMFLOG(BMF_INFO) << "DEBUG, scheduler queue " << id_ << " wait time " << duration << ", wait cnt " << wait_cnt_;
        return 0;
    }

END_BMF_ENGINE_NS

