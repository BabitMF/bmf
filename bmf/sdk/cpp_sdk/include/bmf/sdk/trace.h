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
#ifndef BMF_TRACE_H
#define BMF_TRACE_H

#include <unistd.h>

#include <set>
#include <map>
#include <queue>
#include <string>
#include <thread>
#include <chrono>
#include <atomic>
#include <climits>
#include <cstring>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <filesystem>
#include <unordered_map>

#include "time.h"
#include <bmf/sdk/log.h>
#include <bmf/sdk/common.h>
#include <bmf/sdk/packet.h>

#include <iostream>
BEGIN_BMF_SDK_NS

/* TraceUserInfo is an unordered map to store user info as string
    Note that the key or value should not contain ":" as it is a reserved
   character

    Example:
        TraceUserInfo info = TraceUserInfo();
        info.set("key_string") = "value";   // String format
        info.set("key_int") = 100;          // Int format
        info.set("key_double") = 8.88;      // Double format

    Currently TraceUserInfo supports std::string, int and double only.
*/
class BMF_API TraceUserInfo {
  public:
    std::string data;
    TraceUserInfo() = default;
    ~TraceUserInfo() = default;

    // Set string value
    void set(const char *key, const char *value) {
        data += ",";
        data += key;
        data += ":0:";
        data += value;
    }

    // Set int value
    void set(const char *key, const int value) {
        data += ",";
        data += key;
        data += ":1:";
        data += std::to_string(value);
    }

    // Set double value
    void set(const char *key, const double value) {
        data += ",";
        data += key;
        data += ":2:";
        data += std::to_string(value);
    }

    // Check if there is no key at all
    bool empty() { return data.length() == 0; }

    std::string get() { return data; }
};

/* For checking whether trace is enabled
    Trace is enabled by setting the environment variable "BMF_TRACE":
        export BMF_TRACE=ENABLE
*/
inline const bool TRACE_ENABLED = getenv("BMF_TRACE");

// Default number of buffers for each of the thread
inline const int TRACE_MAX_THREADS = std::thread::hardware_concurrency();

// Default buffer size for each TraceBuffer
inline const int TRACE_MAX_BUFFER_SIZE = 1024;

// Get the buffer size to create TraceBuffer
inline int trace_get_buffer_size() {
    if (getenv("BMF_TRACE_BUFFER_SIZE")) {
        return strtoll(getenv("BMF_TRACE_BUFFER_SIZE"), NULL, 10);
    }
    return TRACE_MAX_BUFFER_SIZE;
}

/* Time interval (in clock cycles) between batched logs
    This ensures logs to be tranched instead of continuous especially for long
   operation
*/
inline int64_t TRACE_BINLOG_INTERVAL =
    1000000 * 10; // 10M clock cycles per binary log

// Configure the logging interval if necessary
inline void BMF_TRACE_SET_BINLOG_INTERVAL(int64_t interval) {
    TRACE_BINLOG_INTERVAL = interval;
}

/* TraceType enum contains the supported trace types
    This includes custom type that can be used by the user
*/
enum TraceType {
    INTERLATENCY,
    PROCESSING,
    SCHEDULE,
    QUEUE_INFO,
    THROUGHPUT,
    CUSTOM,
    TRACE_START,
    GRAPH_START
};

/* TracePhase enum contains the phases for each of the trace event
    For duration event: START, END
    For instant event: NONE
*/
enum TracePhase { NONE, START, END };

// Get all trace types allowed for filtering
inline uint16_t get_trace_allowed() {
    uint16_t allowed_types = 0;

    if (TRACE_ENABLED) {
        if (std::strcmp(getenv("BMF_TRACE"), "ENABLE") != 0) {
            // Since it is not set to ENABLE, it means user had explicitly
            // defined one or more modes that should be enabled
            allowed_types = 0; // Default all modes are disabled until enabled

            // Parse and set the modes chosen by user
            std::stringstream ss(getenv("BMF_TRACE"));

            while (ss.good()) {
                std::string mode;
                getline(ss, mode, ',');

                // Enable recognized mode
                if (mode == "INTERLATENCY") {
                    allowed_types |= (1 << INTERLATENCY);
                } else if (mode == "PROCESSING") {
                    allowed_types |= (1 << PROCESSING);
                } else if (mode == "SCHEDULE") {
                    allowed_types |= (1 << SCHEDULE);
                } else if (mode == "QUEUE_INFO") {
                    allowed_types |= (1 << QUEUE_INFO);
                } else if (mode == "THROUGHPUT") {
                    allowed_types |= (1 << THROUGHPUT);
                } else if (mode == "CUSTOM") {
                    allowed_types |= (1 << CUSTOM);
                } else if (mode == "TRACE_START") {
                    allowed_types |= (1 << TRACE_START);
                }
            }
            // In the event that the variable does not contain any of the
            // aforementioned modes, all allowed types are blacklisted and
            // ultimately filtered out when emitting trace events

        } else {
            // Since it is set to ENABLE, do a blanket enable of all the trace
            // modes (assuming there is less or equals to 8 modes, per bit
            // position = up to 8 bit)
            allowed_types = 0xFF;
        }
    }

    return allowed_types;
}

/* All trace types allowed for filtering

    This can be set using environment variable, for single type:
        export BMF_TRACE=PROCESSING
    For multiple types, separate using comma:
        export BMF_TRACE=PROCESSING,INTERLATENCY

    By setting this, tracing is automatically enabled
*/
inline const uint16_t TRACE_ALLOWED_TYPES = get_trace_allowed();

/* Add or remove trace during compilation
    Include -DNO_TRACE to totally remove trace
*/
#ifndef NO_TRACE

// Get time stamp in microseconds.
inline uint64_t BMF_TRACE_CLOCK() {
    // Using steady_clock for duration measurement
    uint64_t us = std::chrono::duration_cast<std::chrono::microseconds>(
                      std::chrono::steady_clock::now().time_since_epoch())
                      .count();
    return us;
}

inline const uint64_t BMF_TRACE_CLOCK_START = BMF_TRACE_CLOCK();

// TraceEvent class encapsulates all the information emitted by trace
class TraceEvent {
  public:
    int64_t timestamp;
    std::string name;
    std::string subname;
    TraceType category;
    TracePhase phase;
    std::string info;
    TraceEvent();

    TraceEvent(int64_t timestamp, const char *name, const char *subname,
               TraceType category, TracePhase phase)
        : timestamp(timestamp), name(name), subname(subname),
          category(category), phase(phase) {}

    TraceEvent(int64_t timestamp, const char *name, const char *subname,
               TraceType category, TracePhase phase, std::string info)
        : timestamp(timestamp), name(name), subname(subname),
          category(category), phase(phase), info(info) {}
};

/* TraceBuffer class is a lock-free circular buffer of fixed size for single
    producer thread and single consumer thread

    It does not rely on mutex, but rather it uses atomic_int as read and
    write pointer to traverse the buffer. It prevents additional write if the
    buffer is already full, i.e. when buffer is filled by
    (TRACE_MAX_BUFFER_SIZE - 1) TraceEvent objects

    Only logging thread will shift the read pointer, while the event producer
    thread will shift the write pointer. Reading and writing simultaneously
    at the same position will never happen, if is_empty and is_full is checked
    during read and write respectively
*/
class TraceBuffer {
  public:
    std::string process_name;
    std::string thread_name;

    TraceBuffer() : buffer_(trace_get_buffer_size()) {}

    // Writes a new event to buffer
    void push_event(const TraceEvent &event);

    // Reads the oldest event from buffer
    TraceEvent pop_event();

    // Get earliest timestamp in the queue
    int64_t get_next_timestamp() const;

    // Checks if the current event is within the current logging interval
    bool is_not_time_to_log(int64_t time_limit);

    // Checks if buffer is empty
    bool is_empty() const;

    // Checks if buffer is full
    bool is_full() const;

    // Checks how many times buffer has overflowed
    int overflowed() const;

    // Check total number of events emitted
    uint64_t total_count() const;

    // Reset total and overflow counter
    void reset_count();

    int get_buffer_size() const { return buffer_.size(); }

  private:
    std::vector<TraceEvent> buffer_;
    std::atomic_int buffer_occupancy_ = 0;
    std::atomic_int next_read_index_ = 0;
    std::atomic_int next_write_index_ = 0;
    std::atomic_int overflow_count_ = 0;
    std::atomic_uint64_t total_count_ = 0;

    // Get the next index in buffer (wraps back to index 0 after the end)
    int get_next_index(int counter) const;
};

/* TraceLogger class is a global singleton that handles the logging of trace
    events on a separate logging thread

    It performs a batched binary logging based on stipulated intervals, and
    consolidates all binary logs into a final formatted log at the end
*/
class BMF_API TraceLogger {
  public:
    static TraceLogger *traceLogger;

    TraceLogger(
        int queue_size,       // size of vectors to initialize
        bool loop_mode = true // enable starting of logging thread immediately
    );

    // Get the instance of TraceLogger
    static TraceLogger *instance() {
        if (!traceLogger) {
            int thread_count = TRACE_MAX_THREADS;
            if (getenv("BMF_TRACE_BUFFER_COUNT")) {
                thread_count =
                    strtoll(getenv("BMF_TRACE_BUFFER_COUNT"), NULL, 10);
            }
            traceLogger = new TraceLogger(thread_count);
        }
        return traceLogger;
    }

    // Starts the logging thread
    void start();

    // Closes the logging thread and creates the final formatted log
    void end();

    // Register a thread to allocate a dedicated buffer
    int register_queue(std::string process_name, std::string thread_name);

    // Notes a closing thread, hence reducing the current number of running
    // threads
    void close_queue(int thread_id);

    // Adds a new event to the allocated buffer based on thread
    void push(int thread_id, TraceEvent &event);

    // Removes the oldest event from the allocated buffer based on thread
    TraceEvent pop(int thread_id);

    // Generate formatted logs
    void format_logs(bool include_info = true);

  private:
    std::string thread_name_;
    std::string process_name_;
    std::ofstream ofs_;
    std::thread file_thread_;
    std::vector<TraceBuffer> queue_map_;
    std::atomic_int thread_count_ = 0;
    std::atomic_int running_count_ = 0;
    bool thread_quit_ = false;
    int log_index_ = 0;
    int next_format_index_ = 0;
    int64_t current_limit_ = TRACE_BINLOG_INTERVAL;
    bool enable_printing = true;
    bool enable_format_log = true;

    // Get the binary log name based on given log index
    std::string get_log_name(int index);

    // Creates a new binary log file (overwrites if already exists)
    void create_log();

    // Close the current binary log file
    void close_log();

    // Polling the buffers and empty them onto binary log file if within the
    // current log file time interval
    // However the chronological order of entries is not guaranteed within
    // the same log file since there is no requirement for it
    void process_queues();

    // Execution loop for logging
    void loop();

    uint64_t get_duration() {
        return BMF_TRACE_CLOCK() - BMF_TRACE_CLOCK_START;
    }

    void show_overflow() {
        for (int i = 0; i < queue_map_.size(); i++) {
            BMFLOG(BMF_INFO) << "Overflowed for Queue " << i << ": "
                             << queue_map_[i].overflowed() << " / "
                             << queue_map_[i].total_count();
        }
    }
};

// Global singleton of the TraceLogger
// inline TraceLogger *TraceLogger::traceLogger;

/* ThreadTrace class collects all emitted trace events and provides the
    interface for event emission

    Thread-local instances of ThreadTrace is created for each thread, and
    destroyed with the thread
*/
class BMF_API ThreadTrace {
  public:
    ThreadTrace();
    ~ThreadTrace();

    // Standard trace invocation for all event types
    void trace(TraceType category, const char *name, TracePhase phase = NONE,
               const char *src = __builtin_FUNCTION());

    // Trace invocation for all event types with user info
    void trace_info(TraceType category, const char *name, TracePhase phase,
                    std::string info, const char *src = __builtin_FUNCTION());

    // Trace invocation for processing type event
    void trace_process(const char *name, const char *subname, TracePhase phase);

    // Trace invocation for interlatency type event
    void trace_latency(int64_t packet, TracePhase phase);

  private:
    int thread_id_;
    std::string thread_name_;
    std::string process_name_;
};

/* ThreadTrace object that is thread-local to capture all emitted trace
    events within the thread
*/
inline thread_local ThreadTrace threadTracer;

// Assign readable name for scheduler
#define GET_SCHEDULER_NAME "Scheduler"

/* Emit a Trace Event
    Examples:
    BMF_TRACE(SCHEDULE, "Node_A")
    BMF_TRACE(PROCESSING, "Module_B", START)
*/
inline void BMF_TRACE(TraceType category, const char *name,
                      TracePhase phase = NONE,
                      const char *src = __builtin_FUNCTION()) {
    if (TRACE_ALLOWED_TYPES >> category & 1) {
        threadTracer.trace(category, name, phase, src);
    }
}

/* Emit a Trace Event with custom user info
    Example:
    BMF_TRACE(CUSTOM, "QUEUE_C", NONE, TraceUserInfo())
*/
inline void BMF_TRACE(TraceType category, const char *name, TracePhase phase,
                      TraceUserInfo info,
                      const char *src = __builtin_FUNCTION()) {
    if (TRACE_ALLOWED_TYPES >> category & 1) {
        threadTracer.trace_info(category, name, phase, info.get(), src);
    }
}

/* Emit a Trace Event with custom serialized user info
    Example:
    BMF_TRACE(CUSTOM, "QUEUE_C", NONE, ",key:0:value")
*/
inline void BMF_TRACE(TraceType category, const char *name, TracePhase phase,
                      std::string info,
                      const char *src = __builtin_FUNCTION()) {
    if (TRACE_ALLOWED_TYPES >> category & 1) {
        threadTracer.trace_info(category, name, phase, info, src);
    }
}

// Calls the thread-local processing-type trace
inline void BMF_TRACE_PROCESS(const char *name, const char *subname,
                              TracePhase phase) {
    if (TRACE_ALLOWED_TYPES >> PROCESSING & 1) {
        threadTracer.trace_process(name, subname, phase);
    }
}

// Calls the queue info trace
inline void BMF_TRACE_QUEUE_INFO(const char *name, int queue_size, int max_size,
                                 const char *src = __builtin_FUNCTION()) {
    if (TRACE_ALLOWED_TYPES >> QUEUE_INFO & 1) {
        TraceUserInfo info = TraceUserInfo();
        info.set("size", queue_size);
        info.set("max", max_size);
        threadTracer.trace_info(QUEUE_INFO, name, NONE, info.get(), src);
    }
}

// Calls the throughput trace
inline void BMF_TRACE_THROUGHPUT(int stream_id, int node_id, int queue_size) {
    if (TRACE_ALLOWED_TYPES >> THROUGHPUT & 1) {
        TraceUserInfo info = TraceUserInfo();
        info.set("size", queue_size);
        threadTracer.trace_info(
            THROUGHPUT, ("Stream" + std::to_string(stream_id)).c_str(), NONE,
            info.get(), ("Node" + std::to_string(node_id)).c_str());
    }
}

// Indicates to the tracer that tracing is completed, to stop logging and
// start aggregating into formatted log
inline void BMF_TRACE_DONE() {
    if (TRACE_ENABLED) {
        TraceLogger::instance()->format_logs();
    }
}

// Initialize trace (denotes the starting time for calculation)
inline void BMF_TRACE_INIT(const char *src = __builtin_FUNCTION()) {
    if (TRACE_ALLOWED_TYPES >> TRACE_START & 1) {
        threadTracer.trace(TRACE_START, "Trace Start", NONE, src);
    }
}

#define TRACELOG_NAME_FORMAT "tracelog_%Y%m%d_%H%M%S.json"
#define TRACELOG_NAME_SIZE 30

// For emitting process start and end within a function
class TraceProcessEmitter {
  public:
    TraceProcessEmitter() = delete;
    TraceProcessEmitter(TraceType category, std::string node_name,
                        const char *src = __builtin_FUNCTION())
        : node_name_(node_name), src_(src), category_(category) {
        BMF_TRACE(category, node_name.c_str(), START, src);
    };
    ~TraceProcessEmitter() {
        BMF_TRACE(category_, node_name_.c_str(), END, src_.c_str());
    };

  private:
    std::string node_name_;
    std::string src_;
    TraceType category_;
};

#else

#define BMF_TRACE(...)
#define BMF_TRACE_PROCESS(...)
#define BMF_TRACE_LATENCY(...)
#define BMF_TRACE_PROCESS(...)
#define BMF_TRACE_QUEUE_INFO(...)
#define BMF_TRACE_DONE()
#define BMF_TRACE_INIT()
#define BMF_TRACE_THROUGHPUT(...)

#endif

END_BMF_SDK_NS

#endif // BMF_TRACE_H