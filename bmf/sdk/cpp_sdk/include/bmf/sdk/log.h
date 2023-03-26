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

// time measurement
typedef std::chrono::high_resolution_clock::time_point TimeVar;
#define DURATION(a) std::chrono::duration_cast<std::chrono::milliseconds>(a).count()
#define TIMENOW() std::chrono::high_resolution_clock::now()


#ifndef BMF_ENABLE_GLOG

#include <hmp/core/logging.h>
#define BMF_DEBUG hmp::logging::Level::debug
#define BMF_INFO hmp::logging::Level::info
#define BMF_WARNING hmp::logging::Level::warn
#define BMF_ERROR hmp::logging::Level::err
#define BMF_FATAL hmp::logging::Level::fatal


#define BMFLOG_SET_LEVEL(log_level) hmp::logging::set_level(log_level)
#define BMFLOG(log_level)  HMP_SLOG_IF(true, log_level, "BMF")
#define BMFLOG_NODE(log_level, node_id)  BMFLOG(log_level)<<"node id:"<<node_id<<" "


#else //BMF_ENABLE_GLOG

#include <glog/logging.h>

#define BMF_DEBUG 5
#define BMF_INFO 4
#define BMF_WARNING 3
#define BMF_ERROR 2
#define BMF_FATAL 1

#define BMFLOG_SET_LEVEL(log_level) google::SetStderrLogging(log_level)
#define BMFLOG(log_level)  VLOG(log_level)
//#define BMFLOG_NODE(log_level,node_id) COMPACT_GOOGLE_LOG_ ## log_level<<"node id:"<<node_id<<" "
#define BMFLOG_NODE(log_level, node_id)  VLOG(log_level)<<"node id:"<<node_id<<" "


#endif //BMF_ENABLE_GLOG


// Check environment variable to configure logging
inline void BMFLOG_CONFIGURE() {
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
        }
        if (log_level != "DISABLE") {
            BMFLOG_SET_LEVEL(level);
        }
    }
}


#endif //BMF_LOG_H
