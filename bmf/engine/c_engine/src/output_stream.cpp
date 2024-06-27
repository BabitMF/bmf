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

#include "../include/output_stream.h"

#include <bmf/sdk/log.h>

BEGIN_BMF_ENGINE_NS
USE_BMF_SDK_NS

MirrorStream::MirrorStream(
    std::shared_ptr<InputStreamManager> input_stream_manager, int stream_id)
    : input_stream_manager_(input_stream_manager), stream_id_(stream_id) {}

OutputStream::OutputStream(int stream_id, std::string const &identifier,
                           std::string const &alias, std::string const &notify)
    : stream_id_(stream_id), identifier_(identifier), alias_(alias),
      notify_(notify),  queue_(std::make_shared<SafeQueue<Packet>>()) {
        queue_->set_identifier(identifier_);
      }

int OutputStream::add_mirror_stream(
    std::shared_ptr<InputStreamManager> input_stream_manager, int stream_id) {
    mirror_streams_.emplace_back(MirrorStream(input_stream_manager, stream_id));
    return 0;
}

int OutputStream::propagate_packets(
    std::shared_ptr<SafeQueue<Packet>> packets) {
    for (auto &s : mirror_streams_) {
        auto copy_queue = std::make_shared<SafeQueue<Packet>>(*packets.get());
        copy_queue->set_identifier(identifier_);
        s.input_stream_manager_->add_packets(s.stream_id_, copy_queue);
    }
    return 0;
}

int OutputStream::add_packets(std::shared_ptr<SafeQueue<Packet>> packets){
    Packet pkt;
    while (packets->pop(pkt)) {
        queue_->push(pkt);
        if (pkt.timestamp() == EOS or pkt.timestamp() == BMF_EOF) {
            /* add EOF pkt for multi downstream node */
            for (size_t i = 1; i < mirror_streams_.size(); i++) {
                queue_->push(Packet::generate_eof_packet());
            }
        }
    }
    return 0;
}

int OutputStream::split_packets() {
    /* Data Splitting(verified) */
    while (!queue_->empty()) {
        Packet pkt;
        if (queue_->pop(pkt)) {
            auto &s = mirror_streams_[stream_index_];
            auto copy_queue = std::make_shared<SafeQueue<Packet>>();
            copy_queue->push(pkt);
            copy_queue->set_identifier(identifier_);
            // BMFLOG(BMF_INFO) << "Node id: " << node_id_
            //                  << "\tStream_index: " << stream_index_ 
            //                  << "\tmirror_streams' size: " << mirror_streams_.size()
            //                  << "\tCount :" << ++cnt;
            /* original code for single node push pkts to input stream */
            s.input_stream_manager_->add_packets(s.stream_id_, copy_queue);
            /* code for multi node output(verified) */
            // s.input_stream_manager_->add_packets(s.stream_id_, copy_queue, node_id_);
            stream_index_ = (stream_index_ + 1) % mirror_streams_.size();
        }
    }
    return 0;
}

int OutputStream::add_upstream_nodes(int node_id) {
    /* TODO: the node_id_ need to be placed right class like OuputStreamManager */
    node_id_ = node_id;
    for (auto &s : mirror_streams_) {
        s.input_stream_manager_->add_upstream_nodes(node_id);
    }
    return 0;
}

END_BMF_ENGINE_NS
