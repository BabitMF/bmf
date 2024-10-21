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

#include <sstream>
#include <type_traits>
#include <fmt/format.h>
#include <hmp/core/macros.h>

namespace hmp {

#define HMP_FMT(...) ::fmt::format(__VA_ARGS__)
#define HMP_INF(...)                                                           \
    ::hmp::logging::_log(::hmp::logging::Level::info, "HMP",                   \
                         HMP_FMT(__VA_ARGS__).c_str())
#define HMP_WRN(...)                                                           \
    ::hmp::logging::_log(::hmp::logging::Level::warn, "HMP",                   \
                         HMP_FMT(__VA_ARGS__).c_str())
#define HMP_DBG(...)                                                           \
    ::hmp::logging::_log(::hmp::logging::Level::debug, "HMP",                  \
                         HMP_FMT(__VA_ARGS__).c_str())
#define HMP_ERR(...)                                                           \
    ::hmp::logging::_log(::hmp::logging::Level::err, "HMP",                    \
                         HMP_FMT(__VA_ARGS__).c_str())
#define HMP_FTL(...)                                                           \
    ::hmp::logging::_log(::hmp::logging::Level::fatal, "HMP",                  \
                         HMP_FMT(__VA_ARGS__).c_str())

#define HMP_REQUIRE(exp, fmt, ...)                                             \
    if (!(exp)) {                                                              \
        ::hmp::logging::dump_stack_trace();                                    \
        throw std::runtime_error(HMP_FMT("require " #exp " at {}:{}, " fmt,    \
                                         __FILE__, __LINE__, ##__VA_ARGS__));  \
    }

#define HMP_SLOG_IF(condition, level, tag)                                     \
    !(condition) ? (void)0                                                     \
                 : hmp::logging::LogMessageVoidify() &                         \
                       hmp::logging::StreamLogger(level, tag).stream()

#define HMP_SLOG_INF(tag) HMP_SLOG_IF(true, ::hmp::logging::Level::info, tag)
#define HMP_SLOG_WRN(tag) HMP_SLOG_IF(true, ::hmp::logging::Level::warn, tag)
#define HMP_SLOG_DBG(tag) HMP_SLOG_IF(true, ::hmp::logging::Level::debug, tag)
#define HMP_SLOG_ERR(tag) HMP_SLOG_IF(true, ::hmp::logging::Level::err, tag)
#define HMP_SLOG_FTL(tag) HMP_SLOG_IF(true, ::hmp::logging::Level::fatal, tag)

namespace logging {

typedef int (*logCallBackFunc)(int loglevel, const char* msg);
extern logCallBackFunc callBackFunc;

struct Level {
    // map from spdlog::level
    enum {
        trace = 0,
        debug = 1,
        info = 2,
        warn = 3,
        err = 4,
        fatal = 5,
        off = 6,
        n_levels
    };
};

HMP_API void _log(int level, const char *tag, const char *msg);

HMP_API void set_level(int level);
HMP_API void set_format(const std::string &fmt);
HMP_API void set_log_callback_func(logCallBackFunc func);
HMP_API void dump_stack_trace(int max = 128);

class HMP_API LogMessageVoidify {
  public:
    LogMessageVoidify() {}
    // This has to be an operator with a precedence lower than << but
    // higher than ?:
    template <typename T> void operator&(T &) {}
};

class HMP_API StreamLogger {
  public:
    struct OStream {
        virtual OStream &operator<<(const std::string &msg) = 0;
        virtual ~OStream() {}

        // const char*
        OStream &operator<<(const char *v) {
            *this << std::string(v);
            return *this;
        }

        // Numbers
        template <typename T>
        typename std::enable_if<std::is_integral<T>::value ||
                                    std::is_floating_point<T>::value,
                                OStream &>::type
        operator<<(const T &v) {
            *this << std::to_string(v);
            return *this;
        }

        // std::endl
        // this is the type of std::cout
        typedef std::basic_ostream<char, std::char_traits<char>> CoutType;

        // this is the function signature of std::endl
        typedef CoutType &(*StandardEndLine)(CoutType &);

        // define an operator<< to take in std::endl
        OStream &operator<<(StandardEndLine manip) { return *this; }
    };

    StreamLogger() = delete;

    StreamLogger(int level, const char *tag);

    ~StreamLogger();

    OStream &stream();

  private:
    OStream *os_;
    int level_;
    const char *tag_;
};

} // namespace logging
} // namespace hmp
