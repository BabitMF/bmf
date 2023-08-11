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

#ifndef BMF_ENGINE_RUNNING_INFO_H
#define BMF_ENGINE_RUNNING_INFO_H

#include <bmf/sdk/common.h>
#include <bmf/sdk/json_param.h>

#include <nlohmann/json_fwd.hpp>

#include <string>
#include <vector>

USE_BMF_SDK_NS

namespace bmf {
typedef struct PacketInfo {
    std::string data_type;
    std::string class_type;
    std::string class_name;
    int64_t timestamp;

    JsonParam jsonify() {
        nlohmann::json ret;
        ret["data_type"] = data_type;
        ret["class_type"] = class_type;
        ret["class_name"] = class_name;
        ret["timestamp"] = timestamp;

        return JsonParam(ret);
    }
} PacketInfo;

typedef struct TaskStreamInfo {
    uint64_t id;
    std::vector<PacketInfo> packets;

    JsonParam jsonify() {
        nlohmann::json ret;
        ret["id"] = id;
        ret["packets"] = nlohmann::json::array();
        for (auto &p : packets)
            ret["packets"].push_back(p.jsonify().json_value_);

        return JsonParam(ret);
    }
} TaskStreamInfo;

typedef struct TaskInfo {
    uint64_t node_id;
    std::string timestamp;
    int32_t priority;
    std::vector<TaskStreamInfo> input_streams;
    std::vector<uint64_t> output_streams;

    JsonParam jsonify() {
        nlohmann::json ret;
        ret["node_id"] = node_id;
        ret["timestamp"] = timestamp;
        ret["priority"] = priority;
        ret["input_streams"] = nlohmann::json::array();
        for (auto &s : input_streams)
            ret["input_streams"].push_back(s.jsonify().json_value_);
        ret["output_streams"] = nlohmann::json::array();
        for (auto s : output_streams)
            ret["output_streams"].push_back(s);

        return JsonParam(ret);
    }
} TaskInfo;

typedef struct SchedulerQueueInfo {
    int32_t id;
    std::string state;
    int64_t started_at;
    size_t queue_size;
    std::vector<TaskInfo> tasks;

    JsonParam jsonify() {
        nlohmann::json ret;
        ret["id"] = id;
        ret["state"] = state;
        ret["started_at"] = started_at;
        ret["queue_size"] = queue_size;
        ret["tasks"] = nlohmann::json::array();
        for (auto &t : tasks)
            ret["tasks"].push_back(t.jsonify().json_value_);

        return JsonParam(ret);
    }
} SchedulerQueueInfo;

typedef struct SchedulerNodeInfo {
    uint64_t id;
    uint64_t last_scheduled_time;
    uint64_t ref_count;

    JsonParam jsonify() {
        nlohmann::json ret;
        ret["id"] = id;
        ret["last_scheduled_time"] = last_scheduled_time;
        ret["ref_count"] = ref_count;

        return JsonParam(ret);
    }
} SchedulerNodeInfo;

typedef struct SchedulerInfo {
    int64_t last_schedule_success_time;
    std::vector<SchedulerNodeInfo> scheduler_nodes;
    std::vector<SchedulerQueueInfo> scheduler_queues;

    JsonParam jsonify() {
        nlohmann::json ret;
        ret["last_schedule_success_time"] = last_schedule_success_time;
        ret["scheduler_nodes"] = nlohmann::json::array();
        for (auto &nd : scheduler_nodes)
            ret["scheduler_nodes"].push_back(nd.jsonify().json_value_);
        ret["scheduler_queues"] = nlohmann::json::array();
        for (auto &q : scheduler_queues)
            ret["scheduler_queues"].push_back(q.jsonify().json_value_);

        return JsonParam(ret);
    }
} SchedulerInfo;

typedef struct InputStreamInfo {
    uint64_t id;
    uint64_t prev_id, nex_id;
    uint64_t max_size;
    uint64_t size;
    std::string name;
    std::vector<PacketInfo> packets;

    JsonParam jsonify() {
        nlohmann::json ret;
        ret["id"] = id;
        ret["prev_id"] = prev_id;
        ret["nex_id"] = nex_id;
        ret["max_size"] = max_size;
        ret["size"] = size;
        ret["name"] = name;
        ret["packets"] = nlohmann::json::array();
        for (auto &p : packets)
            ret["packets"].push_back(p.jsonify().json_value_);

        return JsonParam(ret);
    }
} StreamInfo;

typedef struct OutputStreamInfo {
    uint64_t id;
    uint64_t prev_id;
    std::string name;
    std::vector<InputStreamInfo> down_streams;

    JsonParam jsonify() {
        nlohmann::json ret;
        ret["id"] = id;
        ret["prev_id"] = prev_id;
        ret["name"] = name;
        ret["down_streams"] = nlohmann::json::array();
        for (auto &s : down_streams)
            ret["down_streams"].push_back(s.jsonify().json_value_);

        return JsonParam(ret);
    }
} OutputStreamInfo;

typedef struct NodeModuleInfo {
    std::string module_name;
    std::string module_type;
    std::string module_entry;
    std::string module_path;

    JsonParam jsonify() {
        nlohmann::json ret;
        ret["module_name"] = module_name;
        ret["module_type"] = module_type;
        ret["module_entry"] = module_entry;
        ret["module_path"] = module_path;

        return JsonParam(ret);
    }
} NodeModuleInfo;

typedef struct NodeRunningInfo {
    uint64_t id;
    std::string type;
    bool is_infinity;
    bool is_source;
    std::string state;
    std::string input_manager_type;
    uint64_t scheduler_queue;
    uint64_t max_pending_task;
    uint64_t pending_task;
    uint64_t task_processed;
    uint64_t schedule_count;
    uint64_t schedule_success_count;
    NodeModuleInfo module_info;
    std::vector<InputStreamInfo> input_streams;
    std::vector<OutputStreamInfo> output_streams;

    JsonParam jsonify() {
        nlohmann::json ret;
        ret["id"] = id;
        ret["is_infinity"] = is_infinity ? "YES" : "NO";
        ret["is_source"] = is_source ? "YES" : "NO";
        ret["state"] = state;
        ret["input_manager_type"] = input_manager_type;
        ret["scheduler_queue"] = scheduler_queue;
        ret["max_pending_task"] = max_pending_task;
        ret["pending_task"] = pending_task;
        ret["task_processed"] = task_processed;
        ret["schedule_count"] = schedule_count;
        ret["schedule_success_count"] = schedule_success_count;
        ret["module_info"] = module_info.jsonify().json_value_;
        ret["input_streams"] = nlohmann::json::array();
        for (auto &s : input_streams)
            ret["input_streams"].push_back(s.jsonify().json_value_);
        ret["output_streams"] = nlohmann::json::array();
        for (auto &s : output_streams)
            ret["output_streams"].push_back(s.jsonify().json_value_);

        return JsonParam(ret);
    }
} NodeRunningInfo;

typedef struct GraphRunningInfo {
    uint64_t id;
    std::string mode;
    std::string state;
    SchedulerInfo scheduler;
    std::vector<std::vector<OutputStreamInfo>> input_streams;
    std::vector<std::vector<InputStreamInfo>> output_streams;
    std::vector<NodeRunningInfo> nodes;

    JsonParam jsonify() {
        nlohmann::json ret;
        ret["id"] = id;
        ret["mode"] = mode;
        ret["state"] = state;
        ret["scheduler"] = scheduler.jsonify().json_value_;
        ret["input_streams"] = nlohmann::json::array();
        for (auto &ss : input_streams) {
            auto tmp = nlohmann::json::array();
            for (auto &s : ss)
                tmp.push_back(s.jsonify().json_value_);
            ret["input_streams"].push_back(tmp);
        }
        ret["output_streams"] = nlohmann::json::array();
        for (auto &ss : output_streams) {
            auto tmp = nlohmann::json::array();
            for (auto &s : ss)
                tmp.push_back(s.jsonify().json_value_);
            ret["output_streams"].push_back(tmp);
        }
        ret["nodes"] = nlohmann::json::array();
        for (auto &nd : nodes)
            ret["nodes"].push_back(nd.jsonify().json_value_);

        return JsonParam(ret);
    }
} GraphRunningInfo;
} // namespace bmf
#endif // BMF_ENGINE_RUNNING_INFO_H
