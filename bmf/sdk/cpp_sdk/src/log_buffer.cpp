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
#include <bmf/sdk/log_buffer.h>

#include <utility>
#include <vector>


namespace bmf_sdk
{
namespace
{

struct LogBufferPrivate
{
    std::mutex mutex;
    std::map<int, std::function<void(std::string const)>> log_cb_hooks;
    int log_cb_idx = 0;
    bool avlog_cb_set = false;
    int avlog_level = 32; //AV_LOG_INFO
    std::map<std::string, short> log_levels;
    void (*av_log_set_callback)(void (*callback)(void *, int, const char *, va_list)) = nullptr;

    LogBufferPrivate()
    {
        log_levels = std::map<std::string, short>{
            {"quiet", -8},   // AV_LOG_QUIET   },
            {"panic", 0},    // AV_LOG_PANIC   },
            {"fatal", 8},    // AV_LOG_FATAL   },
            {"error", 16},   // AV_LOG_ERROR   },
            {"warning", 24}, // AV_LOG_WARNING },
            {"info", 32},    // AV_LOG_INFO    },
            {"verbose", 40}, // AV_LOG_VERBOSE },
            {"debug", 48},   // AV_LOG_DEBUG   },
            {"trace", 56},   // AV_LOG_TRACE   },
        };
    }

    static LogBufferPrivate& inst()
    {
        static LogBufferPrivate p;
        return p;
    }
};

#define self LogBufferPrivate::inst()

} // namespace

void LogBuffer::register_av_log_set_callback(void *func)
{
    std::lock_guard l(self.mutex);
    self.av_log_set_callback = decltype(self.av_log_set_callback)(func);

    if (self.log_cb_hooks.size() > 0) {
        set_av_log_callback();
    }
}

int LogBuffer::set_cb_hook(std::function<void(std::string const)> cb)
{
    std::lock_guard<std::mutex> _(self.mutex);
    if (!self.avlog_cb_set)
        set_av_log_callback();
    self.log_cb_hooks[self.log_cb_idx] = std::move(cb);
    return self.log_cb_idx++;
}

void LogBuffer::remove_cb_hook(int idx)
{
    std::lock_guard<std::mutex> _(self.mutex);
    self.log_cb_hooks.erase(idx);
}

void LogBuffer::lb_callback(void *ptr, int level, const char *fmt, va_list vl)
{
    std::lock_guard<std::mutex> _(self.mutex);
    if (level > self.avlog_level)
        return;

    char message[1024];
    vsnprintf(message, 1023, fmt, vl);
    std::string msg = message;
    for (auto &cb : self.log_cb_hooks)
        cb.second(msg);
}

void LogBuffer::set_av_log_callback()
{
    //std::lock_guard<std::mutex> _(self.mutex);
    if (!self.avlog_cb_set && self.av_log_set_callback != nullptr)
    {
        self.av_log_set_callback(lb_callback);
        self.avlog_cb_set = true;
    }
}

LogBuffer::LogBuffer(std::vector<std::string> &log_buffer)
{
    hook_idx = set_cb_hook([&log_buffer](std::string const log) -> void
                           { log_buffer.push_back(log); });
}

LogBuffer::LogBuffer(std::function<void(const std::string)> log_callback, std::string level)
{
    if (self.log_levels.count(level) > 0)
        self.avlog_level = self.log_levels[level];
    hook_idx = set_cb_hook(log_callback);
}

void LogBuffer::close()
{
    remove_cb_hook(hook_idx);
}


bool LogBuffer::avlog_cb_set()
{
    std::lock_guard<std::mutex> _(self.mutex);
    return self.avlog_cb_set;
}


int LogBuffer::infer_level(const std::string &level_name)
{
    return self.log_levels[level_name];
}

LogBuffer::~LogBuffer()
{
    close();
}
} //namespace bmf_sdk
