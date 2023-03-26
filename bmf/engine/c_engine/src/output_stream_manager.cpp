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

#include "../include/output_stream_manager.h"

#include <bmf/sdk/log.h>

#include <algorithm>
#include <memory>
#include <string>

BEGIN_BMF_ENGINE_NS
    USE_BMF_SDK_NS

    OutputStreamManager::OutputStreamManager(std::vector<StreamConfig> output_streams) {
        // Rearrange 'video' and 'audio' streams to 1st & 2nd positions.
        // Notice that if 'video' or 'audio' appears at nor 1st or 2nd stream, it will not be rearranged.
        if (output_streams.size() > 1 && output_streams[1].get_notify() == "video")
            std::swap(output_streams[0], output_streams[1]);

        // Handling the situation that only an 'audio' stream exists, will force it to be the 2nd stream.
        // An ugly patch, waiting for a better solution.
        if (output_streams.size() ==1 && output_streams[0].get_notify() == "audio"){
            output_streams_[1] = std::make_shared<OutputStream>(0, output_streams[0].get_identifier(),
                                                                output_streams[0].get_alias(),
                                                                output_streams[0].get_notify());
            stream_id_list_.push_back(1);
            return;
        }

        for (int i = 0; i < output_streams.size(); i++) {
            output_streams_[i] = std::make_shared<OutputStream>(i, output_streams[i].get_identifier(),
                                                                output_streams[i].get_alias(),
                                                                output_streams[i].get_notify());

            stream_id_list_.push_back(i);
        }

    }

    int OutputStreamManager::post_process(Task &task) {
        for (auto &t:task.outputs_queue_) {
            auto q = std::make_shared<SafeQueue<Packet> >(t.second);
            output_streams_[t.first]->propagate_packets(q);
        }
        return 0;
    }

    bool OutputStreamManager::get_stream(int stream_id, std::shared_ptr<OutputStream> &output_stream) {
        if (output_streams_.count(stream_id) > 0) {
            output_stream = output_streams_[stream_id];
            return true;
        }
        return false;
    }

    int OutputStreamManager::propagate_packets(int stream_id, std::shared_ptr<SafeQueue<Packet>> packets) {
        output_streams_[stream_id]->propagate_packets(packets);
        return 0;
    }

    bool OutputStreamManager::any_of_downstream_full() {
        for (auto &out_s : output_streams_) {
            for (auto &mirror_stream:(out_s.second->mirror_streams_)) {
                std::shared_ptr<InputStream> downstream;
                mirror_stream.input_stream_manager_->get_stream(mirror_stream.stream_id_, downstream);
                if (downstream->is_full()) {
                    return true;
                }
            }
        }
        return false;
    }

    std::vector<int> OutputStreamManager::get_stream_id_list() {
        return stream_id_list_;
    }

    int OutputStreamManager::add_stream(std::string name) {
        int stream_id;

        max_id_ += 1;
        stream_id = max_id_;

        output_streams_[stream_id] = std::make_shared<OutputStream>(stream_id, name);
        stream_id_list_.push_back(stream_id);

        return stream_id;
    }

    void OutputStreamManager::remove_stream(int stream_id, int mirror_id) {
        int pos = -1;
        for (int i = 0; i < output_streams_[stream_id]->mirror_streams_.size(); i++) {
            auto &mirror_stream = output_streams_[stream_id]->mirror_streams_[i];
            if (mirror_id == -1 || mirror_stream.stream_id_ == mirror_id) {//-1 means all the mirror will be removed
                pos = i;
                mirror_stream.input_stream_manager_->remove_stream(mirror_stream.stream_id_);
                break;
            }
        }

        auto &mirror_s = output_streams_[stream_id]->mirror_streams_;

        if (mirror_id != -1 && pos != -1) {
            mirror_s.erase(mirror_s.begin() + pos);
            BMFLOG(BMF_INFO) << "output stream manager erase mirror id: " << mirror_id << " in stream: "
                             << stream_id;
        }
        if (mirror_id == -1 || mirror_s.size() == 0) {
            output_streams_.erase(stream_id);
            int i;
            for (i = 0; i < stream_id_list_.size(); i++) {
                if (stream_id_list_[i] == stream_id)
                    break;
            }
            stream_id_list_.erase(stream_id_list_.begin() + i);
        }
        return;
    }

    void OutputStreamManager::wait_on_stream_empty(int stream_id) {
        for (auto mirror_stream:output_streams_[stream_id]->mirror_streams_)
            mirror_stream.input_stream_manager_->wait_on_stream_empty(mirror_stream.stream_id_);
    }

    void OutputStreamManager::probe_eof() {
        for (auto output_stream_iter = output_streams_.begin();
             output_stream_iter != output_streams_.end(); output_stream_iter++) {
            for (auto mirror_stream:(output_stream_iter->second->mirror_streams_)) {
                std::shared_ptr<InputStream> downstream;
                mirror_stream.input_stream_manager_->get_stream(mirror_stream.stream_id_, downstream);
                downstream->probe_eof(true);
            }
        }
        return;
    }

    int OutputStreamManager::get_outlink_nodes_id(std::vector<int> &nodes_id) {
        std::map<int, bool> outlink_id;
        for (auto output_stream_iter = output_streams_.begin();
             output_stream_iter != output_streams_.end(); output_stream_iter++) {
            for (auto mirror_stream:(output_stream_iter->second->mirror_streams_))
                outlink_id[mirror_stream.input_stream_manager_->node_id_] = true;
        }
        for (auto it:outlink_id)
            nodes_id.push_back(it.first);
        return 0;
    }
END_BMF_ENGINE_NS
