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
#include <hmp/core/logging.h>

#if defined(__ANDROID__)
#include <android/log.h>
#include <sstream>
#include <iomanip>
#include <unwind.h>
#include <vector>
#include <dlfcn.h>
#if __ANDROID_API__ < 30
#include <atomic>
#endif

#elif defined(__APPLE__)

#include <sstream>
#include <iomanip>
#include <unwind.h>
#include <vector>
#include <dlfcn.h>
#include <atomic>

#else 
#include <spdlog/spdlog.h>
#include <backward.hpp>

#endif //__android__

namespace hmp { namespace logging{

class OStreamImpl : public StreamLogger::OStream
{
    std::stringstream ss_;
public:
    OStream& operator<<(const std::string &msg) override
    {
        ss_ << msg;
        return *this;
    }

    std::string str()
    {
        return ss_.str();
    }

};


StreamLogger::StreamLogger(int level, const char *tag)
        : level_(level), tag_(tag)
{
    os_ = new OStreamImpl();
}

StreamLogger::~StreamLogger()
{
    auto os = static_cast<OStreamImpl*>(os_);
    ::hmp::logging::_log(level_, tag_, os->str().c_str());
    delete os_;
}

StreamLogger::OStream& StreamLogger::stream()
{
    return *os_;
}

#ifdef __ANDROID__
android_LogPriority to_android_priority(int level)
{
    android_LogPriority prio;
    switch(level){
        case Level::trace:
            prio = ANDROID_LOG_VERBOSE; break;
        case Level::debug:
            prio = ANDROID_LOG_DEBUG; break;
        case Level::info:
            prio = ANDROID_LOG_INFO; break;
        case Level::warn:
            prio = ANDROID_LOG_WARN; break;
        case Level::err:
            prio = ANDROID_LOG_ERROR; break;
        case Level::fatal:
            prio = ANDROID_LOG_FATAL; break;
        case Level::off:
            prio = ANDROID_LOG_SILENT; break;
        default:
            prio = ANDROID_LOG_UNKNOWN;
    }
    return prio;
}

#if __ANDROID_API__ < 30

static std::atomic<int> _s_log_prio = ANDROID_LOG_DEFAULT;
int32_t __android_log_set_minimum_priority(android_LogPriority prio)
{
    _s_log_prio = prio;
    return 0;
}
#endif

#elif defined(__APPLE__) //
static std::atomic<int> _s_log_prio = Level::info;
#endif


void set_level(int level)
{
#if defined(__ANDROID__)
    auto prio = to_android_priority(level);
    __android_log_set_minimum_priority(prio);
#elif defined(__APPLE__)
    _s_log_prio = level;
#else
    spdlog::set_level((spdlog::level::level_enum)level);
#endif
}

void set_format(const std::string &fmt)
{
#if !defined(__ANDROID__) && !defined(__APPLE__)
    spdlog::set_pattern(fmt);
#endif
}


void _log(int level, const char* tag, const char *msg)
{
#if defined(__ANDROID__)
    auto prio = to_android_priority(level);
#if __ANDROID_API__ < 30
    if(prio >= _s_log_prio)
#endif
    __android_log_write(prio, tag, msg);

#elif defined(__APPLE__)
    if(level < _s_log_prio){
        return;
    }

    const char *level_name = nullptr;
    switch(level){
        case Level::trace:
            level_name = "TRACE"; break;
        case Level::debug:
            level_name = "DEBUG"; break;
        case Level::info:
            level_name = "INFO"; break;
        case Level::warn:
            level_name = "WARN"; break;
        case Level::err:
            level_name = "ERROR"; break;
        case Level::fatal:
            level_name = "FATAL"; break;
        case Level::off:
            level_name = "OFF"; break;
        default:
            level_name = "UNKNOWN";
    }

    //
    time_t now;
    struct tm *tm_now = nullptr;
    time(&now);
    tm_now = localtime(&now);
    char time_str[128];
    if(tm_now != nullptr){
        snprintf(time_str, sizeof(time_str), "%02d-%02d-%02d %02d:%02d:%02d",
            tm_now->tm_year, tm_now->tm_mon, tm_now->tm_mday, tm_now->tm_hour, 
            tm_now->tm_min, tm_now->tm_sec);
    }
    //
    fprintf(stderr, "%s [%s][%s] %s\n", time_str, level_name, tag, msg);
#else
    spdlog::default_logger_raw()->log(spdlog::source_loc{},
                                      (spdlog::level::level_enum)level,
                                      msg);
#endif
}


#if defined(__ANDROID__) || defined(__APPLE__)

namespace {

//from stackoverflow
struct BacktraceState
{
    void** current;
    void** end;
};

static _Unwind_Reason_Code unwindCallback(struct _Unwind_Context* context, void* arg)
{
    BacktraceState* state = static_cast<BacktraceState*>(arg);
    uintptr_t pc = _Unwind_GetIP(context);
    if (pc) {
        if (state->current == state->end) {
            return _URC_END_OF_STACK;
        } else {
            *state->current++ = reinterpret_cast<void*>(pc);
        }
    }
    return _URC_NO_REASON;
}


size_t captureBacktrace(void** buffer, size_t max)
{
    BacktraceState state = {buffer, buffer + max};
#if defined(__arm64__)
    _Unwind_Backtrace(unwindCallback, &state);
#endif
    return state.current - buffer;
}

void dumpBacktrace(std::ostream& os, void** buffer, size_t count)
{
    for (size_t idx = 0; idx < count; ++idx) {
        const void* addr = buffer[idx];
        const char* symbol = "";

        Dl_info info;
        if (dladdr(addr, &info) && info.dli_sname) {
            symbol = info.dli_sname;
        }

        os << "  #" << std::setw(2) << idx << ": " << addr << "  " << symbol << "\n";
    }
}

} //namespace


void dump_stack_trace(int max)
{
    std::vector<void*> buffer(max);
    std::ostringstream oss;

    logging::dumpBacktrace(oss, buffer.data(),
                           logging::captureBacktrace(buffer.data(), max));

    HMP_WRN("{}", oss.str());
}

#else 

void dump_stack_trace(int max)
{
    using namespace backward;

    StackTrace st; 
    st.load_here(max);

    size_t depth_no_py = max;
    TraceResolver tr;
    tr.load_stacktrace(st);
    for (size_t i = 0; i < st.size(); ++i){
        ResolvedTrace trace = tr.resolve(st[i]);
        if(trace.object_function.substr(0, 3) == "_Py"){
            depth_no_py = i + 1;
            break;
        }
    }

    //
    if(depth_no_py < max){
        std::cerr << "## Python Stack ignored" << std::endl;
        st.load_here(depth_no_py);
    }
    Printer p;
    p.print(st);
}

#endif


}} //namespace hmp::logging
