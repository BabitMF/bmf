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
#include <bmf/sdk/bmf_stat.h>
#include <bmf/sdk/log.h>

namespace bmf_sdk {

BMFStat &BMFStat::GetInstance() {
    static BMFStat bmfst;
    return bmfst;
}

BMFStat::BMFStat() {
    std::string path = ""; // linux, win, ios, android
    upload_lib =
        SharedLibrary(path, SharedLibrary::LAZY | SharedLibrary::GLOBAL);
    if (upload_lib.is_open())
        BMFLOG(BMF_INFO) << "BMF stat upload lib was found and loaded";
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
}

void BMFStat::push_info(JsonParam info) { stat_info.merge_patch(info); }

int BMFStat::upload_info() {
    if (upload_lib.is_open()) {
        // send back the stat info base on the lib for different OS/environments
    }
    return 0;
}

void BMFStat::push_track_point(std::shared_ptr<TrackPoint> track_point) {
    std::lock_guard<std::mutex> lk(mutex_);
    queue_.push(track_point);
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

}

void bmf_stat_report(std::shared_ptr<TrackPoint> track_point) {
    auto &stat = BMFStat::GetInstance();
    stat.push_track_point(track_point);
}

// int64_t BMFStat::task_id() {
//     return task_id_;
// }

} // namespace bmf_sdk
