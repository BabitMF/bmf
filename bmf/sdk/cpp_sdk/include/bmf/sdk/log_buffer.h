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

#ifndef BMF_LOG_BUFFER_H
#define BMF_LOG_BUFFER_H

#include <bmf/sdk/common.h>

#include <functional>
#include <mutex>
#include <map>
#include <vector>
#include <string>

BEGIN_BMF_SDK_NS

class BMF_API LogBuffer {
  public:
    static void set_av_log_callback();
    static void register_av_log_set_callback(void *func);

    LogBuffer(std::vector<std::string> &log_buffer);

    LogBuffer(std::function<void(std::string const)> log_callback,
              std::string level);

    ~LogBuffer();

    void close();

    static bool avlog_cb_set();

    static int infer_level(const std::string &level_name);

  private:
    static int set_cb_hook(std::function<void(std::string const)> cb);

    static void remove_cb_hook(int idx);

    static void lb_callback(void *ptr, int level, char const *fme, va_list vl);

    int hook_idx;
};

END_BMF_SDK_NS
#endif // BMF_LOG_BUFFER_H
