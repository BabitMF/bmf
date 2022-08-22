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
        scheduler_queue_callback.exception_ = [this](int node_id) -> int {
            int scheduler_queue_id;
            std::shared_ptr<Node> node = NULL;
            std::shared_ptr<SchedulerQueue> scheduler_queue;
            this->callback_.get_node_(node_id, node);
            if (!node) {
                BMFLOG(BMF_ERROR) << "node id incorrect in schedule:" << node_id;
                return -1;
            }
            scheduler_queue_id = node->get_scheduler_queue_id();
            scheduler_queue = this->scheduler_queues_[scheduler_queue_id];
            if (scheduler_queue->exception_catch_flag_) {
                exception_flag_ = true;
                eptr_ = scheduler_queue->eptr_;
            }
            for (int i = 0; i < this->scheduler_queues_.size(); i++)
                this->scheduler_queues_[i]->exception_catch_flag_ = true;

            this->callback_.close_report_(node_id, true);
            return 0;
        };
        for (int i = 0; i < scheduler_cnt; i++) {
            std::shared_ptr<SchedulerQueue> scheduler_queue = std::make_shared<SchedulerQueue>(
                    i, scheduler_queue_callback);
            scheduler_queues_.push_back(scheduler_queue);
        }
    }

    int Scheduler::start() {
        for (int i = 0; i < scheduler_queues_.size(); i++) {
            scheduler_queues_[i]->start();
        }
        return 0;
    }

    int Scheduler::close() {
        for (int i = 0; i < scheduler_queues_.size(); i++) {
            scheduler_queues_[i]->close();
        }
        BMFLOG(BMF_INFO) << "all scheduling threads were joint";
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
        bool notify_needed = false;
        if (nodes_to_schedule_.size() == 0)
            notify_needed = true;
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
                //printf("DEBUG, node %d refcnt is: %d\n", node_id, nodes_to_schedule_[node_id].nodes_ref_cnt_);
            } else {
                if (nodes_to_schedule_.count(node_id) > 0) {
                    if (nodes_to_schedule_[node_id].nodes_ref_cnt_ > 0) {
                        nodes_to_schedule_[node_id].nodes_ref_cnt_--;
                    }
                    //printf("DEBUG, node %d refcnt is: %d\n", node_id, nodes_to_schedule_[node_id].nodes_ref_cnt_);
                    if (nodes_to_schedule_[node_id].nodes_ref_cnt_ == 0) {
                        nodes_to_schedule_.erase(node_id);
                    }
                }
            }
        }
        //printf("DEBUG: nodes to scheduler size: %d\n", nodes_to_schedule_.size());

        if (is_add) {
            int64_t start_time = clock();
            std::shared_ptr<Node> sched_node = NULL;
            choose_node_schedule(start_time, sched_node);
            if (sched_node &&
                ((sched_node->is_source() && !sched_node->any_of_downstream_full()) ||
                 (!sched_node->is_source() && !sched_node->too_many_tasks_pending()))) {
                //if (nodes_to_schedule_.size() > 0 && notify_needed) {
                //printf("DEBUG: add node %d and notify all, sched list size: %d\n", sched_node->get_id(), nodes_to_schedule_.size());
                sched_node->pre_sched_num_++;
                //sched_nodes_.push(sched_node);
                to_schedule_queue(sched_node);
                //std::lock_guard<std::mutex> lk(sched_mutex_);
                //sched_needed_.notify_one();
                //}
            }
        }
        node_mutex_.unlock();
        return 0;
    }

    int Scheduler::sched_required(int node_id, bool is_closed) {
        NodeItem final_node_item = NodeItem();
        bool got_node = false;
        int scheduler_queue_id;
        std::shared_ptr<Node> node = NULL;
        std::shared_ptr<SchedulerQueue> scheduler_queue;

        callback_.get_node_(node_id, node);
        if (!node) {
            BMFLOG(BMF_ERROR) << "node id incorrect in schedule:" << node_id;
            return -1;
        }
        if (exception_flag_)
            return 0;
        if (is_closed) {
            callback_.close_report_(node_id, false);
        } else {
            std::shared_ptr<InputStreamManager> input_stream_manager;
            node->get_input_stream_manager(input_stream_manager);
            for (auto &node_id:input_stream_manager->upstream_nodes_)
                sched_required(node_id, false);

            std::lock_guard<std::mutex> lk(node->sched_mutex_);
            if ((!node->too_many_tasks_pending() && !node->any_of_downstream_full())) {
                node->pre_sched_num_++;
                to_schedule_queue(node);
            }
        }
        return 0;
    }

    bool Scheduler::choose_node_schedule(int64_t start_time, std::shared_ptr<Node> &node) {
        node_mutex_.lock();
        NodeItem final_node_item = NodeItem();
        int node_id = -1;
        for (auto node_item:nodes_to_schedule_) {
            //if (node_item.second.node_->pre_sched_num_ <= 4) {
                if (node_item.second.node_->is_source() && node_item.second.node_->any_of_downstream_full() && node_item.second.node_->too_many_tasks_pending()) {
                    if (node_item.second.node_->any_of_downstream_full())
                        printf("DEBUG, node %d, choose the source node which is downstream full\n", node_item.first);
                    if (node_item.second.node_->too_many_tasks_pending())
                        printf("DEBUG, node %d, choose the source node which is pending full\n", node_item.first);
                    node_id = -1;
                    continue;
                }
                if (!node_item.second.node_->is_source() && node_item.second.node_->too_many_tasks_pending() && node_item.second.node_->all_input_queue_empty()) {
                    node_id = -1;
                    continue;
                }
            //} else {
            //    printf("DEBUG, node %d pre sched number is full\n", node_item.first);
            //    continue;
            //}
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
            final_node_item.last_scheduled_time_ = clock();
            node = final_node_item.node_;
            node_mutex_.unlock();
            return true;
        }
        node_mutex_.unlock();
        return false;
    }

    int Scheduler::to_schedule_queue(std::shared_ptr<Node> node) {

        if (node && node->wait_pause_) {
            //break;
            return 0;
        }
        if (node) {
            node->pre_sched_num_--;
            if (node->schedule_node()) {
                last_schedule_success_time_ = clock();
            }
        }
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
