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

#include "../include/input_stream_manager.h"
#include "../include/node.h"

#include <bmf/sdk/common.h>
#include <bmf/sdk/log.h>
#include <bmf/sdk/timestamp.h>

#include <cmath>
#include <memory>

BEGIN_BMF_ENGINE_NS
    USE_BMF_SDK_NS

    InputStreamManager::InputStreamManager(int node_id, std::vector<StreamConfig> &input_streams,
                                           std::vector<int> &output_stream_id_list, uint32_t max_queue_size,
                                           InputStreamManagerCallBack &callback)
            : node_id_(node_id), callback_(callback), output_stream_id_list_(output_stream_id_list) {
        std::function<void(int, bool)> empt;// no need to send callback to input streamer now
        for (auto i = 0; i < input_streams.size(); ++i) {
            input_stream_names_.push_back(input_streams[i].get_identifier());
            input_streams_[i] = std::make_shared<InputStream>(i, input_streams[i], node_id, empt, max_queue_size);
            stream_id_list_.push_back(i);
        }
        max_id_ = input_streams.size() - 1;
    }


    bool InputStreamManager::get_stream(int stream_id, std::shared_ptr<InputStream> &input_stream) {
        if (input_streams_.count(stream_id) > 0) {
            input_stream = input_streams_[stream_id];
            return true;
        }
        return false;
    }

    int InputStreamManager::add_stream(std::string name, int id) {
        int stream_id;

        max_id_ += 1;
        stream_id = max_id_;

        int max_queue_size = 5;
        input_streams_[stream_id] = std::make_shared<InputStream>(stream_id, name, "", "", id,
                                                                  callback_.sched_required, max_queue_size);
        //list add to ensure the queue can be picked up into the task
        stream_id_list_.push_back(stream_id);

        return stream_id;
    }

    int InputStreamManager::remove_stream(int stream_id) {
        std::lock_guard<std::mutex> _(mtx_);

        std::shared_ptr<InputStream> input_stream = input_streams_[stream_id];

        input_stream->wait_on_empty();
        input_streams_.erase(stream_id);
        int i;
        for (i = 0; i < stream_id_list_.size(); i++) {
            if (stream_id_list_[i] == stream_id)
                break;
        }
        stream_id_list_.erase(stream_id_list_.begin() + i);

        if (stream_done_.find(stream_id) != stream_done_.end())
            stream_done_.erase(stream_id);

        return 0;
    }

    int InputStreamManager::wait_on_stream_empty(int stream_id) {
        std::shared_ptr<InputStream> input_stream = input_streams_[stream_id];

        input_stream->wait_on_empty();
        return 0;
    }

    bool InputStreamManager::schedule_node() {
        int64_t min_timestamp;
        NodeReadiness node_readiness = get_node_readiness(min_timestamp);
        if (node_readiness == NodeReadiness::READY_FOR_PROCESS) {

            Task task = Task(node_id_, stream_id_list_, output_stream_id_list_);
            task.set_timestamp(min_timestamp);

            bool result = fill_task_input(task);
            if (not result) {
                BMFLOG_NODE(BMF_INFO, node_id_) << "Failed to fill packet to task";
                return false;
            }
            callback_.scheduler_cb(task);
            //TODO update node schedule_node_success cnt
            //remove to node to add
            return true;

        }
        return false;
    }

    void InputStreamManager::add_packets(int stream_id, std::shared_ptr<SafeQueue<Packet> > packets) {
        // immediately return when node is closed
        // graph_output_stream has no node_callback
        if (callback_.node_is_closed_cb != NULL && callback_.node_is_closed_cb()) {
            return;
        }
        // won't need to notify if empty pkt was passed in order to save schedule cost
        if (packets->size() == 0)
            return;
        if (input_streams_.count(stream_id) > 0) {
            //bool is_empty = input_streams_[stream_id]->is_empty();
            input_streams_[stream_id]->add_packets(packets);
            if (callback_.sched_required != NULL) {
                //if (this->type() != "Immediate" || (this->type() == "Immediate" && is_empty))
                    //callback_.notify_cb();
                    callback_.sched_required(node_id_, false);
            }
        }
    }

    Packet InputStreamManager::pop_next_packet(int stream_id, bool block) {
        if (input_streams_.count(stream_id)) {
            auto stream = input_streams_[stream_id];
            return stream->pop_next_packet(block);
        } else return Packet(0);
    }

    int InputStreamManager::add_upstream_nodes(int node_id) {
        upstream_nodes_.insert(node_id);
        return 0;
    }

    void InputStreamManager::remove_upstream_nodes(int node_id) {
        upstream_nodes_.erase(node_id);
    }

    bool InputStreamManager::find_upstream_nodes(int node_id) {
        return upstream_nodes_.find(node_id) != upstream_nodes_.end();
    }

    ImmediateInputStreamManager::ImmediateInputStreamManager(int node_id, std::vector<StreamConfig> &input_streams,
                                                             std::vector<int> &output_stream_id_list,
                                                             uint32_t max_queue_size,
                                                             InputStreamManagerCallBack &callback)
            : InputStreamManager(node_id, input_streams, output_stream_id_list, max_queue_size, callback) {
        next_timestamp_ = 1;
    }

    std::string ImmediateInputStreamManager::type() {
        return "Immediate";
    }

    int64_t ImmediateInputStreamManager::get_next_timestamp() {
        std::lock_guard<std::mutex> _(mtx_);
        next_timestamp_++;
        return next_timestamp_;
    }

    NodeReadiness ImmediateInputStreamManager::get_node_readiness(int64_t &min_timestamp) {
        for (auto &input_stream : input_streams_) {
            if (not input_stream.second->is_empty()) {
                min_timestamp = get_next_timestamp();
                return NodeReadiness::READY_FOR_PROCESS;
            }
        }
        return NodeReadiness::NOT_READY;
    }

    bool ImmediateInputStreamManager::fill_task_input(Task &task) {
        bool task_filled = false;
        for (auto & input_stream : input_streams_) {
            if (input_stream.second->is_empty()) {
//            task.fill_input_packet(iter->second->get_id(), Packet());
                continue;
            }

            while (not input_stream.second->is_empty()) {
                Packet pkt = input_stream.second->pop_next_packet(false);
                if (pkt.timestamp() == BMF_EOF) {
                    if (input_stream.second->probed_) {
                        BMFLOG(BMF_INFO) << "immediate sync got EOF from dynamical update";
                        pkt.set_timestamp(DYN_EOS);
                        input_stream.second->probed_ = false;
                    } else
                        stream_done_[input_stream.first] = 1;
                }
                task.fill_input_packet(input_stream.second->get_id(), pkt);
                task_filled = true;
            }
        }

        if (stream_done_.size() == input_streams_.size()) {
            task.set_timestamp(BMF_EOF);
        }
        return task_filled;
    }

    DefaultInputManager::DefaultInputManager(int node_id, std::vector<StreamConfig> &input_streams,
                                             std::vector<int> &output_stream_id_list, uint32_t max_queue_size,
                                             InputStreamManagerCallBack &callback)
            : InputStreamManager(node_id, input_streams, output_stream_id_list, max_queue_size, callback) {}

    std::string DefaultInputManager::type() {
        return "Default";
    }

    NodeReadiness DefaultInputManager::get_node_readiness(int64_t &min_timestamp) {
        int64_t min_bound = DONE;
        min_timestamp = DONE;
        for (auto &input_stream:input_streams_) {
            int64_t min_stream_timestamp;
            if (input_stream.second->get_min_timestamp(min_stream_timestamp))
                min_bound = std::min(min_bound, min_stream_timestamp);
            min_timestamp = std::min(min_timestamp, min_stream_timestamp);
        }
        if (min_timestamp == DONE)
            return NodeReadiness::READY_FOR_CLOSE;
        if (min_bound > min_timestamp)
            return NodeReadiness::READY_FOR_PROCESS;
        return NodeReadiness::NOT_READY;
    }

    bool DefaultInputManager::fill_task_input(Task &task) {
        for (auto &input_stream:input_streams_) {
            auto pkt = input_stream.second->pop_packet_at_timestamp(task.timestamp());
            if (pkt.timestamp() == UNSET) {
                continue;
            }
            if (not task.fill_input_packet(input_stream.second->get_id(), pkt))
                return false;
        }
        return true;
    }

    ServerInputStreamManager::ServerInputStreamManager(int node_id, std::vector<StreamConfig> &input_streams,
                                                       std::vector<int> &output_stream_id_list, uint32_t max_queue_size,
                                                       InputStreamManagerCallBack &callback) : InputStreamManager(
            node_id, input_streams, output_stream_id_list, max_queue_size, callback) {
        next_timestamp_ = 1;
        for (auto &stream_key:input_streams_)
            stream_eof_[stream_key.second] = 0;
    }

    std::string ServerInputStreamManager::type() {
        return "Server";
    }

    int64_t ServerInputStreamManager::get_next_time_stamp() {
        std::lock_guard<std::mutex> _(mtx_);
        next_timestamp_++;
        return next_timestamp_;
    }

    NodeReadiness ServerInputStreamManager::get_node_readiness(int64_t &next_timestamp) {
        for (auto input_stream:input_streams_) {
            if (!input_stream.second->is_empty()) {
                next_timestamp = get_next_time_stamp();
                return NodeReadiness::READY_FOR_PROCESS;
            }
        }
        return NodeReadiness::NOT_READY;
    }

    int ServerInputStreamManager::get_positive_stream_eof_number() {
        int cnt = 0;
        for (auto input_stream:stream_eof_)
            if (input_stream.second > 0)
                cnt++;
        return cnt;
    }

    void ServerInputStreamManager::update_stream_eof() {
        for (auto &input_stream:stream_eof_)
            input_stream.second--;
    }

    bool ServerInputStreamManager::fill_task_input(Task &task) {
        for (auto input_stream:input_streams_) {
            if (input_stream.second->get_block())
                continue;
            if (input_stream.second->is_empty())
                continue;
            while (!input_stream.second->is_empty()) {
                auto pkt = input_stream.second->pop_next_packet(false);
                if (pkt.timestamp() == BMF_EOF) {
                    input_stream.second->set_block(true);
                    stream_eof_[input_stream.second]++;
                    task.fill_input_packet(input_stream.second->get_id(), pkt);
                    break;
                } else if (pkt.timestamp() == EOS) {
                    stream_done_[input_stream.first] = 1;
                    break;
                }
                task.fill_input_packet(input_stream.second->get_id(), pkt);
            }
        }
        if (get_positive_stream_eof_number() == input_streams_.size()) {
            update_stream_eof();
            task.set_timestamp(BMF_EOF);
        }
        if (stream_done_.size() == input_streams_.size())
            task.set_timestamp(EOS);
        return true;
    }

    FrameSyncInputStreamManager::FrameSyncInputStreamManager(int node_id, std::vector<StreamConfig> &input_streams,
                                                             std::vector<int> &output_stream_id_list,
                                                             uint32_t max_queue_size,
                                                             InputStreamManagerCallBack &callback)
            : InputStreamManager(node_id, input_streams, output_stream_id_list, max_queue_size, callback) {
        frames_ready_ = false;
        sync_level_ = input_streams_.begin()->first;
        next_timestamp_ = 1;
    }

    std::string FrameSyncInputStreamManager::type() {
        return "FrameSync";
    }

    NodeReadiness FrameSyncInputStreamManager::get_node_readiness(int64_t &min_timestamp) {
        std::lock_guard<std::mutex> _(mtx_);

        while (!frames_ready_) {
            int nb_miss = 0;
            for (auto input_stream:input_streams_) {
                if (have_next_.count(input_stream.first) <= 0) {
                    have_next_[input_stream.first] = false;
                    sync_frm_state_[input_stream.first] = UNSET;
                    pkt_ready_[input_stream.first] = std::make_shared<SafeQueue<Packet>>();
                }
                if (have_next_[input_stream.first] || sync_frm_state_[input_stream.first] == BMF_EOF)
                    continue;

                if (input_stream.second->is_empty()) {
                    if (sync_frm_state_[input_stream.first] != BMF_PAUSE)
                        nb_miss++;

                    continue;
                } else {
                    auto pkt = input_stream.second->pop_next_packet(false);

                    if (pkt.timestamp() == BMF_EOF || pkt.timestamp() == DYN_EOS) {
                        sync_frm_state_[input_stream.first] = BMF_EOF;
                        if (sync_level_ == input_stream.first) {
                            auto pos = input_streams_.find(input_stream.first);
                            while (true) {
                                pos++;
                                if (pos == input_streams_.end())
                                    break;

                                if (sync_frm_state_[pos->first] == BMF_EOF)
                                    continue;
                                else {
                                    sync_level_ = pos->first;
                                    break;
                                }
                            }
                        }
                    } else if (pkt.timestamp() == BMF_PAUSE) {
                        sync_frm_state_[input_stream.first] = BMF_PAUSE;
                        continue;
                    } else if (sync_frm_state_[input_stream.first] == BMF_PAUSE)
                        sync_frm_state_[input_stream.first] = UNSET;

                    next_pkt_[input_stream.first] = pkt;
                    timestamp_next_[input_stream.first] = pkt.timestamp();
                    have_next_[input_stream.first] = true;
                }
            }

            if (nb_miss)
                return NodeReadiness::NOT_READY;

            int64_t timestamp = INT64_MAX;
            for (auto &input_stream:input_streams_) {
                if (have_next_[input_stream.first] && timestamp_next_[input_stream.first] < timestamp)
                    timestamp = timestamp_next_[input_stream.first];
            }

            for (auto &input_stream:input_streams_) {
                if (have_next_[input_stream.first] && timestamp == timestamp_next_[input_stream.first]) {
                    curr_pkt_[input_stream.first] = next_pkt_[input_stream.first];
                    timestamp_[input_stream.first] = timestamp_next_[input_stream.first];
                    timestamp_next_[input_stream.first] = INT64_MAX;
                    have_next_[input_stream.first] = false;

                    if (input_stream.first == sync_level_) {
                        frames_ready_ = true;
                    }
                }
            }
        }

        if (frames_ready_) {
            for (auto &input_stream:input_streams_) {
                if (sync_frm_state_[input_stream.first] != BMF_PAUSE) {
                    curr_pkt_[input_stream.first].set_timestamp(timestamp_[sync_level_]);
                    pkt_ready_[input_stream.first]->push(curr_pkt_[input_stream.first]);
                }
            }
            frames_ready_ = false;
            next_timestamp_++;
            min_timestamp = next_timestamp_;
            return NodeReadiness::READY_FOR_PROCESS;
        } else
            return NodeReadiness::NOT_READY;
    }

    bool FrameSyncInputStreamManager::fill_task_input(Task &task) {
        std::lock_guard<std::mutex> _(mtx_);

        for (auto iter = input_streams_.begin(); iter != input_streams_.end(); iter++) {
            if (stream_done_.find(iter->first) != stream_done_.end())
                continue;

            Packet pkt;
            if (pkt_ready_[iter->second->get_id()]->pop(pkt) == true) {
                if (pkt.timestamp() == BMF_EOF) {
                    if (iter->second->probed_) {
                        pkt.set_timestamp(DYN_EOS);
                        iter->second->probed_ = false;
                    } else
                        stream_done_[iter->first] = 1;
                }

                task.fill_input_packet(iter->second->get_id(), pkt);
            }
        }
        if (stream_done_.size() == input_streams_.size()) {
            task.set_timestamp(BMF_EOF);
        }
        return true;
    }

    ClockBasedSyncInputStreamManager::ClockBasedSyncInputStreamManager(int node_id,
                                                                       std::vector<StreamConfig> &input_streams,
                                                                       std::vector<int> &output_stream_id_list,
                                                                       uint32_t max_queue_size,
                                                                       InputStreamManagerCallBack &callback)
            : InputStreamManager(node_id, input_streams, output_stream_id_list, max_queue_size, callback) {
        zeropoint_offset_ = -1;
        first_frame_offset_ = -1;
        // Ignore the clock stream (aka. stream[0]).
        for (auto input_stream:input_streams_) {
            //skip the 0 stream which is clock by default
            if (input_stream.first == 0)
                continue;
            cache_[input_stream.first] = std::queue<Packet>();
            last_pkt_[input_stream.first] = Packet(0);
            last_pkt_[input_stream.first].set_timestamp(UNSET);
        }
        next_timestamp_ = 1;
    }

    std::string ClockBasedSyncInputStreamManager::type() {
        return "ClockBasedSync";
    }

    NodeReadiness ClockBasedSyncInputStreamManager::get_node_readiness(int64_t &min_timestamp) {
        std::lock_guard<std::mutex> _(mtx_);

        for (auto input_stream:input_streams_) {
            if (input_stream.first == 0)
                continue;
            if (stream_done_.count(input_stream.first) > 0)
                continue;
            while (!input_stream.second->is_empty()) {
                auto pkt = input_stream.second->pop_next_packet(false);
                //if engine probe it, timestamp should be set to DYN_EOS even it's EOF to avoid downstream EOF
                if (pkt.timestamp() == BMF_EOF && input_streams_[input_stream.first]->probed_) {
                    BMFLOG(BMF_INFO) << "sync got EOF from dynamical update";
                    pkt.set_timestamp(DYN_EOS);
                    input_streams_[input_stream.first]->probed_ = false;

                    //pop all the old data to pass the DYN_EOS at the shortest time before the stream was removed
                    while (!cache_[input_stream.first].empty())
                        cache_[input_stream.first].pop();
                    cache_[input_stream.first].push(pkt);
                } else
                    cache_[input_stream.first].push(pkt);
            }
        }
        // Wait a frame gap to get more available frames.
        if (input_streams_[0]->queue_->size() < 2) {
            return NodeReadiness::NOT_READY;
        }

        next_timestamp_++;
        min_timestamp = next_timestamp_;
        return NodeReadiness::READY_FOR_PROCESS;
    }

    bool ClockBasedSyncInputStreamManager::fill_task_input(Task &task) {
        std::lock_guard<std::mutex> _(mtx_);

        auto ts = input_streams_[0]->pop_next_packet(false).timestamp();
        if (zeropoint_offset_ < 0) {
            for (auto &s:cache_)
                if (!s.second.empty()) {
                    auto tmp = s.second.front().timestamp();
                    if (tmp == BMF_EOF || tmp == BMF_PAUSE)
                        continue;
                    first_frame_offset_ = tmp;
                    break;
                }
            if (first_frame_offset_ >= 0)
                zeropoint_offset_ = ts;
        }
        int64_t target_pts = (ts - zeropoint_offset_) + first_frame_offset_;

        for (auto &s:cache_) {
            // If EOF met, ignore all packets after that.
            if (stream_done_.count(s.first)) {
                while (!s.second.empty())
                    s.second.pop();
                continue;
            }
            // Pick the packet closest to the target pts.
            int64_t offset = LLONG_MAX;
            int64_t cal_offset;
            while (!s.second.empty()) {
                auto cur_ts = s.second.front().timestamp();
                // When EOF met, shut down stream immediately.
                if (cur_ts == BMF_EOF) {
                    last_pkt_[s.first] = s.second.front();
                    s.second.pop();
                    break;
                }
                // PAUSE have higher priority and should be treated at once.
                if (cur_ts != BMF_PAUSE) {
                    cal_offset = std::abs(cur_ts - target_pts);
                    if (cal_offset > offset)
                        break;
                    offset = cal_offset;
                }
                last_pkt_[s.first] = s.second.front();
                s.second.pop();
            }
        }

        for (auto &s:last_pkt_) {
            if (stream_done_.count(s.first))
                continue;

            auto pkt = s.second;
            // Ignore empty stream or just resumed stream.
            if (pkt.timestamp() == UNSET)
                continue;

            if (pkt.timestamp() == BMF_EOF) {// || pkt.get_timestamp() == DYN_EOS) {
                stream_done_[s.first] = 1;
            } else if (pkt.timestamp() == BMF_PAUSE) {
                // Pass PAUSE only once.
                if (stream_paused_.count(s.first))
                    continue;
                stream_paused_.insert(s.first);
            } else {
                if (stream_paused_.count(s.first))
                    stream_paused_.erase(s.first);
                if (ts != BMF_EOF)
                    pkt.set_timestamp(target_pts);
            }
            task.fill_input_packet(s.first, pkt);
        }
        if (ts == BMF_EOF || stream_done_.size() == input_streams_.size()) {
            task.set_timestamp(BMF_EOF);
        }

        // Clock stream index is 0 by hard code now. Erase it in task directly.
        task.get_inputs().erase(0);

        return true;
    }

    int create_input_stream_manager(
            std::string const &manager_type, int node_id, std::vector<StreamConfig> input_streams,
            std::vector<int> output_stream_id_list, InputStreamManagerCallBack callback,
            uint32_t queue_size_limit, std::shared_ptr<InputStreamManager> &input_stream_manager) {
        if (manager_type == "immediate") {
            input_stream_manager = std::make_shared<ImmediateInputStreamManager>
                    (node_id, input_streams, output_stream_id_list, queue_size_limit, callback);
        } else if (manager_type == "default") {
            input_stream_manager = std::make_shared<DefaultInputManager>(node_id, input_streams, output_stream_id_list,
                                                                         queue_size_limit, callback);
        } else if (manager_type == "server") {
            input_stream_manager = std::make_shared<ServerInputStreamManager>(node_id, input_streams,
                                                                              output_stream_id_list, queue_size_limit,
                                                                              callback);
        } else if (manager_type == "framesync") {
            input_stream_manager = std::make_shared<FrameSyncInputStreamManager>(node_id, input_streams,
                                                                                 output_stream_id_list,
                                                                                 queue_size_limit, callback);
        } else if (manager_type == "clocksync") {
            input_stream_manager = std::make_shared<ClockBasedSyncInputStreamManager>(node_id, input_streams,
                                                                                      output_stream_id_list,
                                                                                      queue_size_limit, callback);
        } else {
            BMFLOG(BMF_WARNING)
                            << "Unknown input_manager for node[" << node_id << "], will use 'default' to initialize.";
            input_stream_manager = std::make_shared<DefaultInputManager>(node_id, input_streams, output_stream_id_list,
                                                                         queue_size_limit, callback);
        }
        return 0;
    }

END_BMF_ENGINE_NS
