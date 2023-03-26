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

#ifndef BMF_ENGINE_RUNNING_INFO_COLLECTOR_H
#define BMF_ENGINE_RUNNING_INFO_COLLECTOR_H

#include "graph.h"
#include "node.h"
#include "scheduler.h"
#include "scheduler_queue.h"
#include "input_stream.h"
#include "output_stream.h"

#include "../../connector/include/running_info.h"

#include <bmf/sdk/common.h>
#include <bmf/sdk/task.h>
#include <bmf/sdk/packet.h>

BEGIN_BMF_ENGINE_NS
    USE_BMF_SDK_NS

    class RunningInfoCollector {
    public:
        bmf::GraphRunningInfo collect_graph_info(Graph *graph);

        bmf::NodeRunningInfo collect_node_info(Node *node, Graph *graph = nullptr);

        bmf::SchedulerQueueInfo collect_scheduler_queue_info(SchedulerQueue *scheduler_q);

        bmf::SchedulerInfo collect_scheduler_info(Scheduler *scheduler);

        bmf::InputStreamInfo collect_input_stream_info(InputStream *stream);

        bmf::OutputStreamInfo collect_output_stream_info(OutputStream *stream);

        bmf::TaskInfo collect_task_info(Task *task);

        bmf::PacketInfo collect_packet_info(Packet *packet);
    };


END_BMF_ENGINE_NS
#endif //BMF_ENGINE_RUNNING_INFO_COLLECTOR_H
