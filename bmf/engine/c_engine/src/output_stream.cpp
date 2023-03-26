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

    MirrorStream::MirrorStream(std::shared_ptr<InputStreamManager> input_stream_manager, int stream_id)
            : input_stream_manager_(input_stream_manager), stream_id_(stream_id) {}

    OutputStream::OutputStream(int stream_id, std::string const &identifier, std::string const &alias,
                               std::string const &notify) : stream_id_(stream_id), identifier_(identifier),
                                                            alias_(alias), notify_(notify) {}

    int OutputStream::add_mirror_stream(std::shared_ptr<InputStreamManager> input_stream_manager, int stream_id) {
        mirror_streams_.emplace_back(MirrorStream(input_stream_manager, stream_id));
        return 0;
    }

    int OutputStream::propagate_packets(std::shared_ptr<SafeQueue<Packet> > packets) {
        for (auto &s:mirror_streams_) {
            auto copy_queue = std::make_shared<SafeQueue<Packet> >(*packets.get());
            copy_queue->set_identifier(identifier_);
            s.input_stream_manager_->add_packets(s.stream_id_, copy_queue);
        }
        return 0;
    }
END_BMF_ENGINE_NS