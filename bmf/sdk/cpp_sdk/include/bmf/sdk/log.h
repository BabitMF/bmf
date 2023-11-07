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

#ifndef BMF_LOG_H
#define BMF_LOG_H

#include <chrono>
#include <mutex>

// time measurement
typedef std::chrono::high_resolution_clock::time_point TimeVar;
#define DURATION(a)                                                            \
    std::chrono::duration_cast<std::chrono::milliseconds>(a).count()
#define TIMENOW() std::chrono::high_resolution_clock::now()

#ifndef BMF_ENABLE_GLOG

#include <hmp/core/logging.h>
#define BMF_DEBUG hmp::logging::Level::debug
#define BMF_INFO hmp::logging::Level::info
#define BMF_WARNING hmp::logging::Level::warn
#define BMF_ERROR hmp::logging::Level::err
#define BMF_FATAL hmp::logging::Level::fatal
#define BMF_DISABLE hmp::logging::Level::off

#define BMFLOG_SET_LEVEL(log_level) hmp::logging::set_level(log_level)
#define BMFLOG_SET_LOG_CALLBACK(log_callback) hmp::logging::set_log_callback_func(log_callback)
#define BMFLOG(log_level) HMP_SLOG_IF(true, log_level, "BMF")
#define BMFLOG_NODE(log_level, node_id)                                        \
    BMFLOG(log_level) << "node id:" << node_id << " "

// Check environment variable to configure logging
inline void configure_bmf_log_level() {
    // Set the log level to display, in terms of criticality
    if (getenv("BMF_LOG_LEVEL")) {
        std::string log_level(getenv("BMF_LOG_LEVEL"));
        int level = BMF_INFO;
        if (log_level == "WARNING") {
            level = BMF_WARNING;
        } else if (log_level == "ERROR") {
            level = BMF_ERROR;
        } else if (log_level == "FATAL") {
            level = BMF_FATAL;
        } else if (log_level == "DISABLE") {
            level = BMF_DISABLE;
        }

        BMFLOG_SET_LEVEL(level);
    }
}

#else // BMF_ENABLE_GLOG

#include <glog/logging.h>

#define VLOG_LEVEL_DEBUG 4

#define BMF_LOG_BMF_INFO LOG(INFO)
#define BMF_LOG_BMF_WARNING LOG(WARNING)
#define BMF_LOG_BMF_ERROR LOG(ERROR)
#define BMF_LOG_BMF_FATAL LOG(FATAL)
#define BMF_LOG_BMF_DEBUG VLOG(VLOG_LEVEL_DEBUG)

#define BMFLOG(log_level) BMF_LOG_##log_level
#define BMFLOG_NODE(log_level, node_id)                                        \
    BMF_LOG_##log_level << "node id:" << node_id << " "

#define BMF_DEBUG 0
#define BMF_INFO 1
#define BMF_WARNING 2
#define BMF_ERROR 3
#define BMF_FATAL 4
#define BMF_DISABLE 5

#define BMFLOG_SET_LEVEL(log_level) glog_set_log_level(log_level)

inline void glog_set_log_level(int log_level) {
    if (log_level == BMF_INFO) {
        FLAGS_minloglevel = google::GLOG_INFO;
    } else if (log_level == BMF_WARNING) {
        FLAGS_minloglevel = google::GLOG_WARNING;
    } else if (log_level == BMF_ERROR) {
        FLAGS_minloglevel = google::GLOG_ERROR;
    } else if (log_level == BMF_FATAL) {
        FLAGS_minloglevel = google::GLOG_FATAL;
    } else if (log_level == BMF_DEBUG) {
        FLAGS_v = VLOG_LEVEL_DEBUG;
    } else if (log_level == BMF_DISABLE) {
        FLAGS_minloglevel = google::GLOG_FATAL + 1;
    }
}

// Check environment variable to configure logging
inline void configure_bmf_log_level() {
    // Set the log level to display, in terms of criticality
    char *log_env = getenv("BMF_LOG_LEVEL");
    if (log_env) {
        std::string log_level(log_env);
        if (log_level == "INFO") {
            FLAGS_minloglevel = google::GLOG_INFO;
        } else if (log_level == "WARNING") {
            FLAGS_minloglevel = google::GLOG_WARNING;
        } else if (log_level == "ERROR") {
            FLAGS_minloglevel = google::GLOG_ERROR;
        } else if (log_level == "FATAL") {
            FLAGS_minloglevel = google::GLOG_FATAL;
        } else if (log_level == "DEBUG") {
            FLAGS_v = VLOG_LEVEL_DEBUG;
        } else if (log_level == "DISABLE") {
            FLAGS_minloglevel = google::GLOG_FATAL + 1;
        }
    }
}

#endif // BMF_ENABLE_GLOG

inline void configure_bmf_log() {
    static std::once_flag log_init_once;
    std::call_once(log_init_once, configure_bmf_log_level);
}

#endif // BMF_LOG_H
