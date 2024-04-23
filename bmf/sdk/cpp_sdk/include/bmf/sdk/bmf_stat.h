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
#pragma once

#include <bmf/sdk/json_param.h>
#include <bmf/sdk/shared_library.h>

#include <cstdint>
#include <string>
#include <thread>
#include <mutex>
#include <queue>
#include <memory>
#include <condition_variable>

BEGIN_BMF_SDK_NS

inline uint64_t bmf_get_time_monotonic() {
    uint64_t us = std::chrono::duration_cast<std::chrono::microseconds>(
                      std::chrono::steady_clock::now().time_since_epoch())
                      .count();
    return us;
}

inline uint64_t bmf_get_time() {
    uint64_t us = std::chrono::duration_cast<std::chrono::microseconds>(
                      std::chrono::system_clock::now().time_since_epoch())
                      .count();
    return us;
}

class TrackPoint {
  public:
    int64_t get_task_id() { return task_id; }
    virtual std::string get_tag() = 0;
    int64_t task_id = -1;
};

struct GraphStartData : public TrackPoint {
    std::string get_tag() { return "GraphStartData"; }
    int64_t start_timestamp = -1;
    std::string version;
    std::string commit;
};

// macro to generate to_json and from_json function. object key is variable name
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(GraphStartData, task_id, start_timestamp,
                                   version)

// exact key value data
struct GraphEndData : public TrackPoint {
    std::string get_tag() { return "GraphEndData"; }
    int64_t start_timestamp = -1;
    int err = -1;
    std::string graph_str;
};

struct ModuleData : public TrackPoint {
    std::string get_tag() { return "ModuleData"; }
    std::string module_name;
    int64_t avg_processing_time = 0;
    int64_t max_processing_time = 0;
    int64_t min_processing_time = INT64_MAX;
    int process_cnts = 0;
};

struct ModuleFFDecoderData {
    std::string get_tag() { return "ModuleFFDecoderData"; }
    int width = 0;
    int height = 0;
    std::string option;
};

class CalcContext {
  public:
    void add_sample();
    int64_t get_avg_value();
    int64_t get_min_value();
    int64_t get_max_value();

  private:
    int num = 0;
    int sum = 0;
    int min = 0;
    int max = 0;
};

class BMF_SDK_API BMFStat {
  public:
    static BMFStat &GetInstance();
    void push_info(JsonParam info);
    int upload_info(); // if the lib was exists, transfer the stat info back
    void dump(std::string filename); // dump to file or print directly
    int64_t task_id();

    void push_track_point(std::shared_ptr<TrackPoint> track_point);
    void process_track_points();
    void report_stat(std::shared_ptr<TrackPoint> point);

  private:
    BMFStat(); // single instance
    ~BMFStat();
    BMFStat(const BMFStat &bmfst) = delete;
    const BMFStat &operator=(const BMFStat &bmfst) = delete;
    JsonParam stat_info;
    SharedLibrary upload_lib;
    // async upload
    std::thread t_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<std::shared_ptr<TrackPoint>> queue_;
    bool thread_quit_ = false;
};

BMF_SDK_API void bmf_stat_report(std::shared_ptr<TrackPoint> track_point);

END_BMF_SDK_NS
