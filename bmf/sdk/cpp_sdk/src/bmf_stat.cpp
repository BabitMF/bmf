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
#ifdef BMF_ENABLE_STAT
#include <bmf/sdk/bmf_stat.h>
#include <bmf/sdk/log.h>

namespace bmf_sdk {

BMFStat &BMFStat::GetInstance() {
    static BMFStat bmfst;
    return bmfst;
}

BMFStat::BMFStat() {
    std::string path = "libbmf_data_reporter.so"; // linux, win, ios, android
    upload_lib =
        SharedLibrary(path, SharedLibrary::LAZY | SharedLibrary::GLOBAL | SharedLibrary::DEEP_BIND);
    if (upload_lib.is_open()) {
        create_bmf_kafka_reporter_default create_func = upload_lib.symbol<create_bmf_kafka_reporter_default>("create_bmf_kafka_reporter_default_online");
        reporter_ = create_func();
        BMFLOG(BMF_INFO) << "BMF stat upload lib was found and loaded";
    }

    
    t_ = std::thread(&BMFStat::process_track_points, this);
}

BMFStat::~BMFStat() {
    {
        std::lock_guard<std::mutex> lk(mutex_);
        thread_quit_ = true;
        cv_.notify_one();
    }

    if (t_.joinable()) {
        t_.join();
    }

    if (reporter_) {
        auto start = std::chrono::steady_clock::now();
        reporter_->flush();
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        BMFLOG(BMF_INFO) << "Produce flush Elapsed time: " << elapsed.count() << " seconds";
        destroy_kafka_reporter del_func = upload_lib.symbol<destroy_kafka_reporter>("destroy_kafka_reporter");
        del_func(reporter_);
    }
}

void BMFStat::push_info(JsonParam info) { stat_info.merge_patch(info); }

int BMFStat::upload_info() {
    if (reporter_) {
        // send back the stat info base on the lib for different OS/environments
    }
    return 0;
}

void BMFStat::push_track_point(std::shared_ptr<TrackPoint> track_point) {
    std::lock_guard<std::mutex> lk(mutex_);
    queue_.push(track_point);
    cv_.notify_one();
}

void BMFStat::process_track_points() {
    while (true) {
        std::shared_ptr<TrackPoint> point;

        {
            std::unique_lock<std::mutex> lock(mutex_);
            if (queue_.empty() && thread_quit_) {
                break;
            } else if (queue_.empty() && !thread_quit_) {
                cv_.wait(lock,
                         [this] { return !queue_.empty() || thread_quit_; });
            }

            if (!queue_.empty()) {
                point = queue_.front();
                queue_.pop();
            }
        }

        if (point)
            report_stat(point);
    }
}

void BMFStat::report_stat(std::shared_ptr<TrackPoint> point) {
    std::string data_tag = point->get_tag();
    std::string data_content = point->to_json().dump();
    bmq_data.outputs[data_tag] = data_content;
    if (data_tag == "GraphEndData" && reporter_) {
        std::string data_tag = bmq_data.get_tag();
        std::string data_content = bmq_data.to_json().dump();
        BMFLOG(BMF_INFO) << "data point tag: " << data_tag;
        BMFLOG(BMF_INFO) << "data point content: " << data_content;
        auto start = std::chrono::steady_clock::now();
        if (reporter_->produce(data_content)) {
            BMFLOG(BMF_INFO) << "msg reported successfully!";
        } else {
            BMFLOG(BMF_ERROR) << "msg reported failed!";
        }
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        BMFLOG(BMF_INFO) << "Produce message Elapsed time: " << elapsed.count() << " seconds";
        bmq_data = BMQData();
    }
}

void bmf_stat_report(std::shared_ptr<TrackPoint> track_point) {
    auto &stat = BMFStat::GetInstance();
    stat.push_track_point(track_point);
}

int64_t BMFStat::task_id() {
    return task_id_;
}

bool BMFStat::set_user_id(int user_id) {
    search_user_id_func search_func = upload_lib.symbol<search_user_id_func>("search_user_id");
    if (search_func(user_id)) {
        user_id_ = user_id;
        return true;
    }
    return false;
}

} // namespace bmf_sdk
#endif