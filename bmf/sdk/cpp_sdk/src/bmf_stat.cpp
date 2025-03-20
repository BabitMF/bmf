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
    try {
        std::string path = "libbmf_data_reporter.so"; // linux, win, ios, android
        upload_lib =
            SharedLibrary(path, SharedLibrary::LAZY | SharedLibrary::GLOBAL);
        if (upload_lib.is_open()) {
            std::string create_func_symbol = "create_bmf_kafka_reporter_default_online";
            if (getenv("BMF_STAT_ENV")) {
                std::string stat_env{getenv("BMF_STAT_ENV")};
                if (stat_env == "BOE") {
                    create_func_symbol = "create_bmf_kafka_reporter_default_boe";
                }
            }
            create_bmf_kafka_reporter_default create_func = upload_lib.symbol<create_bmf_kafka_reporter_default>(create_func_symbol);
            reporter_ = create_func();
            BMFLOG(BMF_INFO) << "BMF stat upload lib was found and loaded";
            t_ = std::thread(&BMFStat::process_track_points, this);
            is_enabled_ = true;
        }
    } catch (const std::runtime_error& e) {
        BMFLOG(BMF_INFO) << "BMF stat upload lib load failed: " << e.what();
    } catch (...) {
        BMFLOG(BMF_INFO) << "BMF stat catch unknown exception";
    }

    if (!is_enabled_) {
        BMFLOG(BMF_INFO) << "BMF stat feature will be closed!";
    }
    
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
        auto start = std::chrono::steady_clock::now();
        std::string data_tag = bmq_data.get_tag();
        std::string data_content = bmq_data.to_json().dump();
        BMFLOG(BMF_DEBUG) << "data point tag: " << data_tag;
        BMFLOG(BMF_DEBUG) << "data point content: " << data_content;
        if (reporter_->produce(data_content)) {
            BMFLOG(BMF_INFO) << "msg reported successfully!";
        } else {
            BMFLOG(BMF_ERROR) << "msg reported failed!";
        }
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        BMFLOG(BMF_INFO) << "Produce message Elapsed time: " << elapsed.count() << " seconds";
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
    if (data_tag == "ModuleData") {
        module_data_list.emplace_back(point);
        return;
    }
    std::string data_content = point->to_json().dump();
    bmq_data.outputs[data_tag] = data_content;
    if (data_tag == "GraphEndData") {
        if (!module_data_list.empty()) {
            nlohmann::json module_data_json = nlohmann::json::array();
            module_data_json.get_ref<nlohmann::json::array_t&>().reserve(module_data_list.size());
            
            for (const auto &module_data : module_data_list) {
                module_data_json.emplace_back(*static_cast<ModuleData*>(module_data.get()));
            }
            
            bmq_data.outputs["ModuleDataList"] = std::move(module_data_json.dump());
        }
        upload_info();
        bmq_data.outputs.clear();
        module_data_list.clear(); 
    }
}

int64_t BMFStat::task_id() {
    return task_id_;
}

void BMFStat::set_user_id(const std::string& user_id) {
    user_id_ = user_id;
}

bool BMFStat::is_enabled() {
    return is_enabled_;
}

void bmf_stat_report(std::shared_ptr<TrackPoint> track_point) {
    auto &stat = BMFStat::GetInstance();
    stat.push_track_point(track_point);
}

bool bmf_stat_enabled() {
    auto &stat = BMFStat::GetInstance();
    return stat.is_enabled();
}



} // namespace bmf_sdk