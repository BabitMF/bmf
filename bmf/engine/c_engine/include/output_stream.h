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

#ifndef BMF_OUTPUT_STREAM_H
#define BMF_OUTPUT_STREAM_H

#include"input_stream_manager.h"
#include<queue>

BEGIN_BMF_ENGINE_NS
    USE_BMF_SDK_NS

    class MirrorStream {
    public:
        MirrorStream(std::shared_ptr<InputStreamManager> input_stream_manager, int stream_id);

        std::shared_ptr<InputStreamManager> input_stream_manager_;
        int stream_id_;
    };

    class OutputStream {
    public:
        OutputStream(int stream_id, std::string const &identifier, std::string const &alias = "",
                     std::string const &notify = "");

        int add_mirror_stream(std::shared_ptr<InputStreamManager> input_stream_manager, int stream_id);

        int propagate_packets(std::shared_ptr<SafeQueue<Packet>> packets);

        int stream_id_;
        std::string identifier_;
        std::string notify_;
        std::string alias_;
        std::vector<MirrorStream> mirror_streams_;
    };
END_BMF_ENGINE_NS

#endif //BMF_OUTPUT_STREAM_H
