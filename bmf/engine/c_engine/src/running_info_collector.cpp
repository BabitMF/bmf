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

#include "../include/running_info_collector.h"
#include "../include/graph.h"

#include "../../connector/include/running_info.h"

#include <queue>

BEGIN_BMF_ENGINE_NS
    USE_BMF_SDK_NS

    bmf::GraphRunningInfo RunningInfoCollector::collect_graph_info(Graph *graph) {
        //graph->pause_running();
        bmf::GraphRunningInfo graph_info;
        // TODO: Not implemented.
        graph_info.id = 0;
        graph_info.mode = [](bmf_sdk::BmfMode mode) -> std::string {
            switch (mode) {
                case bmf_sdk::BmfMode::NORMAL_MODE:
                    return "Normal";
                case bmf_sdk::BmfMode::SERVER_MODE:
                    return "Server";
                case bmf_sdk::BmfMode::GENERATOR_MODE:
                    return "Generator";
                default:
                    return "UNKNOWN";
            }
        }(graph->graph_config_.get_mode());
        // TODO: Not implemented.
        graph_info.state = "UNKNOWN";
        graph_info.scheduler = collect_scheduler_info(graph->scheduler_.get());

        for (auto &s:graph->input_streams_) {
            std::vector<bmf::OutputStreamInfo> tmp;
            for (auto &os:s.second->manager_->output_streams_)
                tmp.push_back(collect_output_stream_info(os.second.get()));
            graph_info.input_streams.push_back(tmp);
        }

        for (auto &s:graph->output_streams_) {
            std::vector<bmf::InputStreamInfo> tmp;
            for (auto &s:s.second->input_manager_->input_streams_)
                tmp.push_back(collect_input_stream_info(s.second.get()));
            graph_info.output_streams.push_back(tmp);
        }

        for (auto &nd:graph->nodes_)
            graph_info.nodes.push_back(collect_node_info(nd.second.get(), graph));

        graph->resume_running();
        return graph_info;
    }

    bmf::SchedulerInfo RunningInfoCollector::collect_scheduler_info(Scheduler *scheduler) {
        bmf::SchedulerInfo scheduler_info;
        scheduler_info.last_schedule_success_time = scheduler->last_schedule_success_time_;

        for (auto &nd:scheduler->nodes_to_schedule_) {
            bmf::SchedulerNodeInfo tmp;
            tmp.id = uint64_t(nd.second.node_->id_);
            tmp.last_scheduled_time = uint64_t(nd.second.last_scheduled_time_);
            tmp.ref_count = uint64_t(nd.second.nodes_ref_cnt_);

            scheduler_info.scheduler_nodes.push_back(tmp);
        }
        for (auto &q:scheduler->scheduler_queues_)
            scheduler_info.scheduler_queues.push_back(collect_scheduler_queue_info(q.get()));

        return scheduler_info;
    }

    bmf::SchedulerQueueInfo RunningInfoCollector::collect_scheduler_queue_info(SchedulerQueue *scheduler_q) {
        bmf::SchedulerQueueInfo sq_info;
        sq_info.id = scheduler_q->id_;
        sq_info.queue_size = scheduler_q->queue_.size();
        // TODO: Not implemented.
        sq_info.started_at = scheduler_q->start_time_;

        switch (scheduler_q->paused_state_) {
            case State::INITED:
                sq_info.state = "INITED";
                break;
            case State::PAUSED:
                sq_info.state = "PAUSED";
                break;
            case State::RUNNING:
                sq_info.state = "RUNNING";
                break;
            case State::TERMINATED:
                sq_info.state = "TERMINATED";
                break;
            case State::TERMINATING:
                sq_info.state = "TERMINATING";
                break;
            default:
                sq_info.state = std::to_string(int64_t(scheduler_q->paused_state_)) + " -> UNKNOWN";
        }
        // PriorityQueue need buffer to store and push back later.
        auto siz = scheduler_q->queue_.size();
        std::queue<Item> buf;
        while (siz--) {
            auto t = scheduler_q->pop_task();
            auto tmp = collect_task_info(&t.task);
            tmp.priority = t.priority;
            buf.push(t);
            sq_info.tasks.push_back(tmp);
        }
        while (!buf.empty()) {
            scheduler_q->add_task(buf.front().task, buf.front().priority);
            buf.pop();
        }

        return sq_info;
    }

    bmf::NodeRunningInfo RunningInfoCollector::collect_node_info(Node *node, Graph *graph) {
        bmf::NodeRunningInfo node_info;
        node_info.id = uint64_t(node->id_);
        node_info.type = node->type_;
        node_info.is_source = node->is_source_;
        node_info.is_infinity = node->infinity_node_;
        bmf::NodeModuleInfo module_info;
        module_info.module_name = node->module_info_.module_name;
        module_info.module_path = node->module_info_.module_path;
        module_info.module_entry = node->module_info_.module_entry;
        module_info.module_type = node->module_info_.module_type;
        node_info.module_info = module_info;
        node_info.max_pending_task = uint64_t(node->max_pending_tasks_);
        node_info.pending_task = uint64_t(node->pending_tasks_);
        node_info.task_processed = uint64_t(node->task_processed_cnt_);
        node_info.input_manager_type = node->node_config_.get_input_manager() + " -> " +
                                       node->input_stream_manager_->type();
        node_info.scheduler_queue = uint64_t(node->scheduler_queue_id_);
        node_info.schedule_count = uint64_t(node->schedule_node_cnt_);
        node_info.schedule_success_count = uint64_t(node->schedule_node_success_cnt_);

        switch (node->state_) {
            case NodeState::NOT_INITED:
                node_info.state = "NOT_INITED";
                break;
            case NodeState::RUNNING:
                node_info.state = "RUNNING";
                break;
            case NodeState::PENDING:
                node_info.state = "PENDING";
                break;
            case NodeState::CLOSED:
                node_info.state = "CLOSED";
                break;
            default:
                node_info.state = std::to_string(int64_t(node->state_)) + " -> UNKNOWN";
        }

        auto f = [](Graph *graph, uint64_t node_id, uint32_t stream_id) -> uint64_t {
            if (graph == nullptr)
                return INT_MAX;
            for (auto &nd:graph->nodes_)
                for (auto &os:nd.second->output_stream_manager_->output_streams_)
                    for (auto &is:os.second->mirror_streams_)
                        if (is.input_stream_manager_->node_id_ == node_id && is.stream_id_ == stream_id)
                            return nd.second->id_;
            return INT_MAX;
        };
        for (auto &s:node->input_stream_manager_->input_streams_) {
            auto tmp = collect_input_stream_info(s.second.get());
            tmp.prev_id = f(graph, node->id_, s.first);
            node_info.input_streams.push_back(tmp);
        }
        for (auto &s:node->output_stream_manager_->output_streams_) {
            auto tmp = collect_output_stream_info(s.second.get());
            tmp.prev_id = node->id_;
            for (auto &s:tmp.down_streams)
                s.prev_id = node->id_;
            node_info.output_streams.push_back(tmp);
        }

        return node_info;
    }

    bmf::InputStreamInfo RunningInfoCollector::collect_input_stream_info(InputStream *stream) {
        bmf::InputStreamInfo s_info;
        s_info.id = uint64_t(stream->stream_id_);
        s_info.prev_id = INT_MAX;
        s_info.nex_id = uint64_t(stream->node_id_);
        s_info.name = stream->identifier_;
        s_info.max_size = uint64_t(stream->max_queue_size_);
        s_info.size = uint64_t(stream->queue_->size());

        auto siz = stream->queue_->size();
        while (siz--) {
            Packet tmp;
            stream->queue_->pop(tmp);
            s_info.packets.push_back(collect_packet_info(&tmp));
            stream->queue_->push(tmp);
        }

        return s_info;
    }

    bmf::OutputStreamInfo RunningInfoCollector::collect_output_stream_info(OutputStream *stream) {
        bmf::OutputStreamInfo s_info;
        s_info.id = uint64_t(stream->stream_id_);
        s_info.name = stream->identifier_;

        for (auto &s:stream->mirror_streams_)
            s_info.down_streams.push_back(
                    collect_input_stream_info(s.input_stream_manager_->input_streams_[s.stream_id_].get()));

        return s_info;
    }

    bmf::TaskInfo RunningInfoCollector::collect_task_info(Task *task) {
        bmf::TaskInfo task_info;
        task_info.node_id = uint64_t(task->node_id_);
        task_info.priority = 0;

        switch (int64_t(task->timestamp_)) {
            case -1:
                task_info.timestamp = "UNSET";
                break;
            case 9223372036854775802:
                task_info.timestamp = "BMF_PAUSE";
                break;
            case 9223372036854775804:
                task_info.timestamp = "BMF_EOF";
                break;
            case 9223372036854775805:
                task_info.timestamp = "EOS";
                break;
            case 9223372036854775806:
                task_info.timestamp = "INF_SRC";
                break;
            case 9223372036854775807:
                task_info.timestamp = "DONE";
                break;
            default:
                task_info.timestamp = std::to_string(int64_t(task->timestamp_));
        }

        for (auto &s:task->outputs_queue_)
            task_info.output_streams.push_back(s.first);
        for (auto &s:task->inputs_queue_) {
            bmf::TaskStreamInfo ts_info = {
                    .id=uint64_t(s.first)
            };
            auto siz = s.second->size();
            while (siz--) {
                auto tmp = s.second->front();
                s.second->pop();
                ts_info.packets.push_back(collect_packet_info(&tmp));
                s.second->push(tmp);
            }
            task_info.input_streams.push_back(ts_info);
        }

        return task_info;
    }

    bmf::PacketInfo RunningInfoCollector::collect_packet_info(Packet *packet) {
        bmf::PacketInfo packet_info;
        packet_info.timestamp = packet->timestamp();
        packet_info.class_name = packet->type_info().name;
        packet_info.class_type = packet->type_info().name;
        packet_info.data_type = "UNSET";

        return packet_info;
    }

END_BMF_ENGINE_NS
