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
#include "../include/input_stream.h"

#include <bmf/sdk/log.h>

#include<iostream>

BEGIN_BMF_ENGINE_NS
    USE_BMF_SDK_NS

    InputStream::InputStream(int stream_id, std::string const &identifier, std::string const &alias,
                             std::string const &notify, int node_id, std::function<void(int, bool)> &throttled_cb,
                             int max_queue_size)
            : stream_id_(stream_id), identifier_(identifier), alias_(alias), notify_(notify),
              node_id_(node_id), throttled_cb_(throttled_cb), max_queue_size_(max_queue_size),
              queue_(std::make_shared<SafeQueue<Packet>>()) {
                  queue_->set_identifier(identifier_);
              }

    InputStream::InputStream(int stream_id, StreamConfig &stream_config, int node_id,
                             std::function<void(int, bool)> &throttled_cb, size_t max_queue_size)
            : stream_id_(stream_id), identifier_(stream_config.get_identifier()), alias_(stream_config.get_alias()),
              notify_(stream_config.get_notify()), node_id_(node_id), throttled_cb_(throttled_cb),
              max_queue_size_(max_queue_size), queue_(std::make_shared<SafeQueue<Packet>>()) {
                  queue_->set_identifier(identifier_);
              }

//InputStream::InputStream(const InputStream &input_stream) {
//    stream_id_ = input_stream.stream_id_;
//    name_ = input_stream.name_;
//    stream_manager_name_ = input_stream.stream_manager_name_;
//    max_queue_size_ = input_stream.max_queue_size_;
//    throttled_cb_ = input_stream.throttled_cb_;
//    node_id_ = input_stream.node_id_;
//}
//
//InputStream &InputStream::operator=(const InputStream &input_stream) {
//    stream_id_ = input_stream.stream_id_;
//    name_ = input_stream.name_;
//    stream_manager_name_ = input_stream.stream_manager_name_;
//    max_queue_size_ = input_stream.max_queue_size_;
//    throttled_cb_ = input_stream.throttled_cb_;
//    node_id_ = input_stream.node_id_;
//}

    int InputStream::add_packets(std::shared_ptr<SafeQueue<Packet> > &packets) {
        Packet pkt;
        while (packets->pop(pkt)) {
            queue_->push(pkt);
            // advance time bounding
            next_time_bounding_ = pkt.timestamp() + 1;
            // if received EOS, set stream done
            // here we can't use pkt.get_timestamp() + 1 since
            // Timestamp.DONE = Timestamp.EOS + 2
            if (pkt.timestamp() == EOS or pkt.timestamp() == BMF_EOF) {
                next_time_bounding_ = DONE;
                BMFLOG_NODE(BMF_INFO, node_id_) << "eof received";
                // * removed previous logic
                // add node to scheduler thread until this EOS is processed
                // graph_output_stream may not have attribute node
                //if (node_id_ >= 0 && throttled_cb_ != NULL) {
                //    throttled_cb_(node_id_, false);
                //}
            }
            // wake up event
            fill_packet_event_.notify_all();
        }
        return 0;
    }

    Packet InputStream::pop_packet_at_timestamp(int64_t timestamp) {
        //TODO return the exactly same timestamp or the most closest one
        Packet pkt;
        Packet temp_pkt;
        while (queue_->front(temp_pkt)) {
            int64_t queue_front_timestamp = temp_pkt.timestamp();
            if (queue_front_timestamp <= timestamp) {
                queue_->pop(pkt);
            } else {
                break;
            }
        }
        if (pkt.timestamp() == EOS or pkt.timestamp() == BMF_EOF) {
            // EOS is popped, remove node from scheduler thread
            BMFLOG_NODE(BMF_INFO, node_id_) << "eof processed, remove node from scheduler";
            //if (node_id_ >= 0)
            //    throttled_cb_(node_id_, false);
        }
        return pkt;
    }

    bool InputStream::is_empty() {
        return queue_->empty();
    }

    void InputStream::wait_on_empty() {
        while (not queue_->empty()) {
            if (next_time_bounding_ == DONE)
                break;
            if (node_id_ >= 0)
                throttled_cb_(node_id_, false);//tobefix
            std::unique_lock<std::mutex> lk(stream_m_);
            stream_ept_.wait_for(lk, std::chrono::microseconds(40));
        }

        return;
    }

    int64_t InputStream::get_time_bounding() {
        return next_time_bounding_;
    }

    int InputStream::get_id() {
        return stream_id_;
    }

    Packet InputStream::pop_next_packet(bool block) {
        Packet pkt;
        if (queue_->pop(pkt)) {
            if (pkt.timestamp() == EOS or pkt.timestamp() == BMF_EOF) {
                // EOS is popped, remove node from scheduler thread
                BMFLOG_NODE(BMF_INFO, node_id_) << "eof processed, remove node from scheduler";
                //if (node_id_ >= 0)
                //    throttled_cb_(node_id_, false);
            }
            return pkt;
        } else {
            std::lock_guard<std::mutex> lk(stream_m_);
            stream_ept_.notify_all();

            if (block) {
                while (queue_->empty()) {
                    //fill_packet_event_.wait(lk);
                    std::unique_lock<std::mutex> lk(mutex_);
                    fill_packet_event_.wait_for(lk, std::chrono::milliseconds(5));
                }
            }
            queue_->pop(pkt);
        }
        return pkt;
    }

    bool InputStream::is_full() {
        return queue_->size() >= max_queue_size_;
    }

    void InputStream::set_connected(bool connected) {
        connected_ = connected;
    }

    bool InputStream::is_connected() {
        return connected_;
    }

    std::string InputStream::get_identifier() {
        return identifier_;
    }

    std::string InputStream::get_alias() {
        return alias_;
    }

    std::string InputStream::get_notify() {
        return notify_;
    }

    void InputStream::clear_queue() {
        Packet pkt;
        while (not queue_->empty()) {
            queue_->pop(pkt);
        }
    }

    bool InputStream::get_min_timestamp(int64_t &min_timestamp) {
        if (queue_->empty()) {
            min_timestamp = next_time_bounding_;
            return true;
        }
        Packet pkt;
        queue_->front(pkt);
        min_timestamp = pkt.timestamp();
        return false;
    }

    bool InputStream::get_block() {
        return block_;
    }

    void InputStream::set_block(bool block) {
        block_ = block;
    }

    void InputStream::probe_eof(bool probed) {
        probed_ = true;
    }
END_BMF_ENGINE_NS
