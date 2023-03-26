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

#include "../include/scheduler.h"
#include "../include/node.h"

#include <bmf/sdk/log.h>

#include <bmf/sdk/trace.h>

BEGIN_BMF_ENGINE_NS
    USE_BMF_SDK_NS

    NodeItem::NodeItem(std::shared_ptr<Node> node) : node_(node) {
        last_scheduled_time_ = 0;
        nodes_ref_cnt_ = 0;
    }

    Scheduler::Scheduler(SchedulerCallBack callback, int scheduler_cnt) {
        thread_quit_ = false;
        callback_ = callback;
        SchedulerQueueCallBack scheduler_queue_callback;
        scheduler_queue_callback.get_node_ = callback.get_node_;
        for (int i = 0; i < scheduler_cnt; i++) {
            std::shared_ptr<SchedulerQueue> scheduler_queue = std::make_shared<SchedulerQueue>(
                    i, scheduler_queue_callback);
            scheduler_queues_.push_back(scheduler_queue);
        }
    }

    int Scheduler::start() {
        exec_thread_ = std::thread(&Scheduler::scheduling_thread, this);
        auto handle = exec_thread_.native_handle();
#if __APPLE__
        pthread_setname_np("scheduler");
#else
        pthread_setname_np(handle,"scheduler");
#endif
        for (int i = 0; i < scheduler_queues_.size(); i++) {
            scheduler_queues_[i]->start();
        }
        return 0;
    }

    int Scheduler::close() {
        if (exec_thread_.joinable()){
            thread_quit_ = true;
            BMFLOG(BMF_INFO) << "closing the scheduling thread";
            exec_thread_.join();
            for (int i = 0; i < scheduler_queues_.size(); i++) {
                scheduler_queues_[i]->close();
            }
            BMFLOG(BMF_INFO) << "all scheduling threads were joint";
        }
        return 0;
    }

    void Scheduler::pause() {
        paused_ = true;
        for (auto &q: scheduler_queues_)
            q->pause();
    }

    void Scheduler::resume() {
        for (auto &q: scheduler_queues_)
            q->resume();
        paused_ = false;
    }

    int Scheduler::add_or_remove_node(int node_id, bool is_add) {
        node_mutex_.lock();
//    BMFLOG(BMF_INFO)<<"add or remove node :"<<node_id<<" add flag:"<<is_add;
        std::shared_ptr<Node> node = NULL;
        callback_.get_node_(node_id, node);
        if (node != NULL) {
            if (is_add) {

                if (nodes_to_schedule_.count(node_id) > 0) {
                    nodes_to_schedule_[node_id].nodes_ref_cnt_++;
                } else {
                    nodes_to_schedule_[node_id] = NodeItem(node);
                    nodes_to_schedule_[node_id].nodes_ref_cnt_++;
                }
            } else {
                if (nodes_to_schedule_.count(node_id) > 0) {
                    if (nodes_to_schedule_[node_id].nodes_ref_cnt_ > 0) {
                        nodes_to_schedule_[node_id].nodes_ref_cnt_--;
                    }
                    if (nodes_to_schedule_[node_id].nodes_ref_cnt_ == 0) {
                        nodes_to_schedule_.erase(node_id);
                    }
                }
            }
        }
        node_mutex_.unlock();
        return 0;
    }

    bool Scheduler::choose_node_schedule(int64_t start_time, std::shared_ptr<Node> &node) {
        node_mutex_.lock();
        NodeItem final_node_item = NodeItem();
        int node_id = -1;
        for (auto node_item:nodes_to_schedule_) {
            if (node_item.second.last_scheduled_time_ <= start_time) {
                if (final_node_item.node_ == NULL) {
                    final_node_item = node_item.second;
                    node_id = node_item.second.node_->get_id();
                } else {
                    if (node_item.second.last_scheduled_time_ < final_node_item.last_scheduled_time_) {
                        final_node_item = node_item.second;
                        node_id = node_item.second.node_->get_id();
                    }
                }
            }
        }

        if (node_id != -1) {
            nodes_to_schedule_[node_id].last_scheduled_time_ = clock();
//        final_node_item.last_scheduled_time_ = clock();
            node = final_node_item.node_;
            node_mutex_.unlock();
            return true;
        }
        node_mutex_.unlock();
        return false;
    }

    int Scheduler::scheduling_thread() {
        bool schedule_success = false;
        bool scheduler_queue_exception_catch_flag_ = false;
        while (not thread_quit_) {
            for(auto scheduler_queue:scheduler_queues_){
                if (scheduler_queue->exception_catch_flag_){
                    scheduler_queue_exception_catch_flag_ = true;
                    break;
                }
            }
            if (scheduler_queue_exception_catch_flag_)
                break;
            if (paused_)
                usleep(1000);
            schedule_success = false;
            int64_t start_time = clock();
            int node_id = -1;
            while (true) {
                if (paused_)
                    break;
                std::shared_ptr<Node> node = NULL;
                bool result = choose_node_schedule(start_time, node);
                if (node && node->wait_pause_) {
                    schedule_success = true;
                    break;
                }
                if (result) {
                    if ((node->is_source() && node->is_hungry()) || (not node->is_source())) {
                        if (node->schedule_node()) {
                            schedule_success = true;
                            last_schedule_success_time_ = clock();
                        }
                        break;
                    }
                } else {
                    break;
                }
            }
            if (not schedule_success) {
                usleep(1000);
            }
        }
        for(auto scheduler_queue:scheduler_queues_){
            if (scheduler_queue->eptr_){
                this->eptr_ = scheduler_queue->eptr_;
                break;
            }
        }
        BMFLOG(BMF_INFO) << "exit the scheduling thread";
        return 0;
    }

    int Scheduler::schedule_node(Task &task) {
        int node_id = task.node_id_;
        std::shared_ptr<Node> node;
        callback_.get_node_(node_id, node);
        node->inc_pending_task();
        std::shared_ptr<SchedulerQueue> scheduler_queue;
        int scheduler_queue_id = node->get_scheduler_queue_id();
        scheduler_queue = scheduler_queues_[scheduler_queue_id];
        // TODO: ??? Useless priority
        scheduler_queue->add_task(task, 1);
        return 0;
    }

    int Scheduler::clear_task(int node_id, int scheduler_queue_id) {
        std::shared_ptr<SchedulerQueue> scheduler_queue = scheduler_queues_[scheduler_queue_id];
        scheduler_queue->remove_node_task(node_id);
        return 0;
    }
END_BMF_ENGINE_NS
