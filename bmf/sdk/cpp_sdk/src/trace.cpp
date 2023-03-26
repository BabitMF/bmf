
#include <bmf/sdk/trace.h>

#include <unistd.h>
#include <bmf_nlohmann/json.hpp>

BEGIN_BMF_SDK_NS

#ifndef NO_TRACE

    TraceEvent::TraceEvent() {}

    void TraceBuffer::push_event(const TraceEvent &event) {
        total_count_++;
        // Handle overflow
        if (is_full()) {
            overflow_count_++;
            // Do not write anymore at limit
            return;
        }
        buffer_[next_write_index_] = event;
        next_write_index_.store(
            get_next_index(next_write_index_), 
            std::memory_order_relaxed
        );
        buffer_occupancy_++;
    }

    TraceEvent TraceBuffer::pop_event() {
        // Assumes that is_empty() has been checked prior to Read
        TraceEvent event = buffer_[next_read_index_];
        next_read_index_.store(
            get_next_index(next_read_index_), 
            std::memory_order_relaxed
        );
        buffer_occupancy_--;
        return event;
    }

    int64_t TraceBuffer::get_next_timestamp() const {
        return buffer_[next_read_index_].timestamp;
    }

    bool TraceBuffer::is_not_time_to_log(int64_t time_limit) {
        return get_next_timestamp() > time_limit;
    }

    bool TraceBuffer::is_empty() const {
        return next_read_index_.load(std::memory_order_relaxed) 
            == next_write_index_.load(std::memory_order_relaxed);
    }

    bool TraceBuffer::is_full() const {
        return get_next_index(next_write_index_.load(std::memory_order_relaxed)) 
            == next_read_index_.load(std::memory_order_relaxed);
    }

    int TraceBuffer::overflowed() const {
        return overflow_count_;
    }

    uint64_t TraceBuffer::total_count() const {
        return total_count_.load(std::memory_order_relaxed); 
    }

    void TraceBuffer::reset_count() {
        overflow_count_.store(0, std::memory_order_relaxed);
        total_count_.store(0, std::memory_order_release);
    }

    int TraceBuffer::get_next_index(int counter) const {
        counter++;
        if (counter >= buffer_.size()) {
            counter = 0;
        }
        return counter;
    }

    TraceLogger::TraceLogger(int queue_size, bool loop_mode) 
        : queue_map_(queue_size) {
        // Set the thread name
        std::thread::id tid = std::this_thread::get_id();
        std::stringstream tss;
        tss << tid;
        thread_name_ = tss.str();

        // Set the process name
        pid_t pid = getpid();
        std::stringstream pss;
        pss << pid;
        process_name_ = pss.str();

        // Set the trace settings
        if (getenv("BMF_TRACE_PRINTING") && std::strcmp(getenv("BMF_TRACE_PRINTING"), "DISABLE") == 0)
            enable_printing = false;
        if (getenv("BMF_TRACE_LOGGING") && std::strcmp(getenv("BMF_TRACE_LOGGING"), "DISABLE") == 0)
            enable_format_log = false;

        // Start the execution loop for logger
        if (loop_mode) {
            start();
        }
    }

    void TraceLogger::start() {
        if (TRACE_ENABLED) {
            file_thread_ = std::thread(&TraceLogger::loop, this);
        }
    }

    void TraceLogger::end() {
        if (TRACE_ENABLED) {
            thread_quit_ = true;
            file_thread_.join();
        }
    }

    int TraceLogger::register_queue(
        std::string process_name,
        std::string thread_name
    ) {
        // Assign buffer for the thread
        queue_map_[thread_count_].process_name = process_name;
        queue_map_[thread_count_].thread_name = thread_name;
        running_count_++;
        if (thread_count_ == queue_map_.size() - 1)
            thread_count_ = 0;      // Back to first buffer to reuse buffer
        return thread_count_++;
    }

    void TraceLogger::close_queue(int thread_id) {
        running_count_--;
        if (!running_count_) {
            end();
        }
    }

    void TraceLogger::push(int thread_id, TraceEvent& event) {
        queue_map_[thread_id].push_event(event);
    }

    TraceEvent TraceLogger::pop(int thread_id) {
        return queue_map_[thread_id].pop_event();
    }

    std::string TraceLogger::get_log_name(int index) {
        return "log" + std::to_string(index) + ".txt";
    }

    void TraceLogger::create_log() {
        ofs_.open(
            get_log_name(log_index_),
            std::ofstream::out | std::ofstream::trunc
        );
    }

    void TraceLogger::close_log() {
        ofs_.close();
        log_index_++;
    }

    void TraceLogger::process_queues() {
        bool break_file = get_duration() > current_limit_;

        for (int i = 0; i < queue_map_.size(); i++) {
            while (!queue_map_[i].is_empty()) {
                TraceEvent event = pop(i);
                ofs_ << queue_map_[i].process_name << "," \
                    << queue_map_[i].thread_name << "," \
                    << event.timestamp << "," \
                    << event.name << ":" << event.subname << "," \
                    << event.category << "," \
                    << event.phase \
                    << event.info \
                    << std::endl;
            }
        }

        // Triggered the splitting of new log file for next batch of events
        if (break_file) {
            close_log();
            current_limit_ += TRACE_BINLOG_INTERVAL;
            create_log();
        }
    }

    void TraceLogger::loop() {
        create_log();

        // Start execute logging
        while (!thread_quit_) {
            process_queues();
            usleep(1);
        }

        // Perform a final clear of pending events to process
        process_queues();

        // Handle log close
        close_log();

        // Print out overflowed buffers
        show_overflow();

        // Format binary logs into JSON log
        BMF_TRACE_DONE();
    }

    ThreadTrace::ThreadTrace() {
        // Check the environment variable "BMF_TRACE", as long as it has been set,
        // trace is enabled. To skip trace, unset "BMF_TRACE"
        if (TRACE_ENABLED) {

            // Set the thread name
            std::thread::id tid = std::this_thread::get_id();
            std::stringstream tss;
            tss << tid;
            thread_name_ = tss.str();

            // Set the process name
            pid_t pid = getpid();
            std::stringstream pss;
            pss << pid;
            process_name_ = pss.str();

            // Register with Tracer
            thread_id_ = TraceLogger::instance()->register_queue(
                process_name_,
                thread_name_
            );
        }
    }
    
    ThreadTrace::~ThreadTrace() {
        if (TRACE_ENABLED) {
            TraceLogger::instance()->close_queue(thread_id_);
        }
    }

    void ThreadTrace::trace(
        TraceType category,
        const char* name,
        TracePhase phase,
        const char* src
    ) {
        // Get the current timestamp
        int64_t timestamp = BMF_TRACE_CLOCK() - BMF_TRACE_CLOCK_START;

        // Create a trace event
        TraceEvent event = TraceEvent(
            timestamp,
            name,
            src,
            category,
            phase
        );

        // Send the event to buffer
        TraceLogger::instance()->push(thread_id_, event);
    }

    void ThreadTrace::trace_info(
        TraceType category,
        const char* name,
        TracePhase phase,
        std::string info,
        const char* src
    ) {
        // Get the current timestamp
        int64_t timestamp = BMF_TRACE_CLOCK() - BMF_TRACE_CLOCK_START;

        // Create a trace event
        TraceEvent event = TraceEvent(
            timestamp,
            name,
            src,
            category,
            phase,
            info
        );

        // Send the event to buffer
        TraceLogger::instance()->push(thread_id_, event);
    }

    void ThreadTrace::trace_process(
        const char* name,
        const char* subname,
        TracePhase phase
    ) {
        // Get the current timestamp
        int64_t timestamp = BMF_TRACE_CLOCK() - BMF_TRACE_CLOCK_START;

        // Create a trace event
        TraceEvent event = TraceEvent(
            timestamp,
            name,
            subname,
            PROCESSING,
            phase
        );

        // Send the event to buffer
        TraceLogger::instance()->push(thread_id_, event);
    }

    void TraceLogger::format_logs(bool include_info) {
        if (!enable_format_log)
            return;

        std::string categories[] = {
            "INTERLATENCY",
            "PROCESSING",
            "SCHEDULE",
            "QUEUE_INFO",
            "THROUGHPUT",
            "CUSTOM",
            "TRACE_START",
            "GRAPH_START"
        };

        std::string phases[] = {
            "i",
            "B",
            "E"
        };

        // Hold the timestamp of duration event with phase 'B'
        std::map<std::string, long long> previous_ts;

        // Count the additional number of duration event with 'B' phase to add
        std::map<std::string, std::map<long long, bool> > ts_completed;

        // Count the occurrence of each event
        std::map<std::string, uint16_t> occurrence;
        std::map<std::string, std::map<std::string, long long> > timings;
        std::map<std::string, std::map<std::string, uint16_t> > queue_info;
        std::map<std::string, std::map<std::string, std::pair<long long, uint16_t> > > throughput;

        bmf_nlohmann::json flog;
        char line[1024];
        long long init_time = LLONG_MAX;
        long long start_time = LLONG_MAX;
        long long final_time = 0;

        for (const auto &entry : std::filesystem::directory_iterator(".")) {
            // Proceed if is a log file
            std::string filename = entry.path().filename().u8string();
            if (filename.size() > 7 && filename.find("log") == 0
                && filename.compare(filename.size() - 4, 4, ".txt") == 0) {
                FILE* fp = fopen(filename.c_str(), "r");

                while (fgets(line, sizeof(line), fp)) {
                    bmf_nlohmann::json linelog;

                    // Line by line decoding
                    std::stringstream ss(line);
                    int term_count = 0;
                    int cat_index = 0;

                    while (ss.good()) {
                        std::string term;
                        getline(ss, term, ',');
                        
                        if (term_count == 0) {
                            // Process ID
                            linelog["pid"] = term;
                        } else if (term_count == 1) {
                            // Thread ID
                            linelog["tid"] = term;
                        } else if (term_count == 2) {
                            // Timestamp
                            char *result;
                            linelog["ts"] = strtoll(term.c_str(), &result, 10);
                        } else if (term_count == 3) {
                            // Name
                            linelog["name"] = term;
                        } else if (term_count == 4) {
                            // Category
                            cat_index = strtoll(term.c_str(), NULL, 10);
                            linelog["cat"] = categories[cat_index];
                            if (cat_index == 6) {
                                // Graph start
                                init_time = linelog["ts"];
                            } else if (enable_printing && cat_index == 4) {
                                // Throughput
                                std::string line_name = linelog["name"];
                                std::string stream_name = line_name.substr(0, line_name.find(":"));
                                std::string node_name = line_name.substr(line_name.find(":") + 1, line_name.length());
                                if (throughput.count(node_name)) {
                                    if (!throughput[node_name].count(stream_name)) {
                                        throughput[node_name][stream_name] = std::make_pair(0, 0);
                                    }
                                    throughput[node_name][stream_name].second += 1;
                                }
                            } else {
                                if (linelog["ts"] < start_time)
                                    start_time = linelog["ts"];
                                if (linelog["ts"] > final_time)
                                    final_time = linelog["ts"];
                            }

                            if (enable_printing && cat_index > 0) {
                                occurrence[linelog["name"]]++;
                            }
                        } else if (term_count == 5) {
                            // Phase
                            linelog["ph"] = phases[
                                strtoll(term.c_str(), NULL, 10)
                            ];

                            // Duration event with phase 'B' will be recorded
                            if (linelog["ph"] == "B") {
                                previous_ts[linelog["name"]] = linelog["ts"];

                                if (enable_printing) {
                                    // throughput 
                                    std::string line_name = linelog["name"];
                                    std::string node_name = line_name.substr(0, line_name.find(":"));
                                    if (!throughput.count(node_name)) {
                                        throughput[node_name]
                                            = std::map<std::string, std::pair<long long, uint16_t> >();
                                    }
                                }
                            } else if (linelog["ph"] == "E") {
                                // Duration event with phase 'E' will be checked against 'E'
                                if (previous_ts.count(linelog["name"])) {
                                    long long last_ts = previous_ts[linelog["name"]];
                                    // Assume 'B' only occur once, while 'E' can occur multiple times
                                    if (ts_completed.count(linelog["name"]) 
                                        && ts_completed[linelog["name"]].count(last_ts)) {
                                        // Duplicate an event for "B" and slot in
                                        bmf_nlohmann::json duplog;
                                        duplog["pid"] = linelog["pid"];
                                        duplog["tid"] = linelog["tid"];
                                        duplog["ts"] = last_ts;
                                        duplog["name"] = linelog["name"];
                                        duplog["cat"] = linelog["cat"];
                                        duplog["ph"] = "B";
                                        flog.push_back(duplog);
                                    } else {
                                        ts_completed[linelog["name"]][last_ts] = true;
                                    }
                                    
                                    // Record the event duration
                                    if (enable_printing && cat_index > 0) {
                                        if (!timings.count(linelog["name"])) {
                                            timings[linelog["name"]]["total"] = 0;
                                            timings[linelog["name"]]["count"] = 0;
                                            timings[linelog["name"]]["ave"] = 0;
                                            timings[linelog["name"]]["min"] = LLONG_MAX;
                                            timings[linelog["name"]]["max"] = 0;
                                        }
                                        long long event_ts = linelog["ts"];
                                        long long duration = event_ts - last_ts;
                                        timings[linelog["name"]]["total"] += duration;
                                        timings[linelog["name"]]["count"]++;
                                        timings[linelog["name"]]["ave"] 
                                            = timings[linelog["name"]]["total"] / timings[linelog["name"]]["count"];
                                        timings[linelog["name"]]["min"]
                                            = std::min(timings[linelog["name"]]["min"], duration);
                                        timings[linelog["name"]]["max"]
                                            = std::max(timings[linelog["name"]]["max"], duration);

                                        std::string line_name = linelog["name"];
                                        std::string node_name = line_name.substr(0, line_name.find(":"));
                                        std::string cat_name = linelog["cat"];
                                        if (cat_index == 1) {
                                            // Handle throughput event - only end of processing will be checked
                                            std::string func_name = line_name.substr(
                                                line_name.find(":") + 1, line_name.length()
                                            );
                                            if (strcmp(func_name.c_str(), "process_node") == 0) {
                                                for (auto &it : throughput[node_name]) {
                                                    it.second.first += duration;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        } else {
                            // Create user info for first occurrence of user info
                            // within the event log
                            if (term_count == 6) {
                                bmf_nlohmann::json user_info;
                                linelog["args"] = user_info;
                            }

                            // Add user info
                            size_t pos = 0;
                            std::string arg, key, valtype;
                            while ((pos = term.find(":")) != std::string::npos) {
                                arg = term.substr(0, pos);
                                if (key.empty()) {
                                    // Handle key
                                    key = arg;
                                } else {
                                    // Handle string type value
                                    valtype = arg;
                                    arg = term.substr(pos + 1, term.length());
                                    if (!arg.empty() && arg[arg.length() - 1] == '\n') {
                                        arg.erase(arg.length() - 1);
                                    }
                                    if (valtype == "1") {
                                        // Handle int type value (decimal string)
                                        linelog["args"][key] = strtoll(arg.c_str(), NULL, 10);
                                    } else if (valtype == "2") {
                                        // Handle double type value (decimal string)
                                        linelog["args"][key] = atof(arg.c_str());
                                    } else {
                                        // Handle string type (default) value
                                        linelog["args"][key] = arg;
                                    }
                                }
                                term.erase(0, pos + 1);
                            }
                        }
                        term_count++;
                    }

                    // Handle queue info
                    if (enable_printing) {
                        if (cat_index == 3) {
                            std::string queue_name = linelog["name"];
                            queue_name = queue_name.substr(0, queue_name.find(":"));
                            if (!queue_info.count(queue_name)) {
                                queue_info[queue_name]["limit"] = linelog["args"]["max"];
                                queue_info[queue_name]["max"] = 0;
                                queue_info[queue_name]["ave"] = 0;
                                queue_info[queue_name]["total"] = 0;
                                queue_info[queue_name]["count"] = 0;
                            }
                            if (linelog["args"]["size"] > queue_info[queue_name]["max"])
                                queue_info[queue_name]["max"] = linelog["args"]["size"];
                            int curr_size = linelog["args"]["size"];
                            queue_info[queue_name]["total"] += curr_size;
                            queue_info[queue_name]["count"]++;
                            queue_info[queue_name]["ave"]
                                = queue_info[queue_name]["total"] / queue_info[queue_name]["count"];
                        }
                    }

                    flog.push_back(linelog);
                }

                fclose(fp);

                // Handle removal of old binary log
                // Binary log that has been formatted is no longer needed
                try {
                    std::filesystem::remove(filename.c_str());
                } catch (const std::filesystem::filesystem_error& err) {
                    std::cerr << "Filesystem Error: "<< err.what() << std::endl;
                }
            }
        }

        // Short-circuit if empty log
        if (!flog.size())
            return;

        // Sort the JSON list
        using json_it_type = decltype(*flog.begin());
        std::sort(flog.begin(), flog.end(), [](const json_it_type a, const json_it_type b) {
            return a["ts"] < b["ts"];
        });

        // Process and print statistics
        if (enable_printing) {
            std::cout << "\n\n- - - - - - - - - - - RUNTIME INFORMATION - - - - - - - - - - - -";

            if (init_time < LLONG_MAX) {
                auto init_diff = final_time - init_time;
                std::cout << "\nTotal execution time (from TRACE_START): " << init_diff << " us";
            }

            auto trace_diff = final_time - start_time;
            std::cout << "\nTotal graph execution time: " << trace_diff << " us";

            std::cout << "\n\n- - - - - - - - - - - - EVENTS FREQUENCY - - - - - - - - - - - - -"
                << std::endl << std::left << std::setw(30) << "Event Name"
                << std::setw(10) << "Total Count";
            std::vector<std::pair<std::string, uint16_t> > sorted_occurrence;
            for (auto &it : occurrence) {
                sorted_occurrence.push_back(it);
            }
            sort(sorted_occurrence.begin(), sorted_occurrence.end(), [](const auto &a, const auto &b) {
                return a.second > b.second;
            });
            for (auto &it : sorted_occurrence) {
                std::cout << std::endl << std::left << std::setw(30) << it.first << std::setw(10) << it.second;
            }

            std::cout << "\n\n- - - - - - - - - - - - EVENTS DURATION - - - - - - - - - - - - -"
                << std::endl << std::left << std::setw(30) << "Event Name"
                << std::setw(10) << "Total (us)"
                << std::setw(10) << "Total (%)"
                << std::setw(10) << "Ave (us)"
                << std::setw(10) << "Min (us)"
                << std::setw(10) << "Max (us)";
            for (auto &it : timings) {
                float perc = it.second["total"] * 100 / trace_diff;
                std::cout << std::endl << std::left << std::setw(30) << it.first
                    << std::setw(10) << it.second["total"]
                    << std::setw(10) << std::fixed << std::setprecision(1) << perc
                    << std::setw(10) << it.second["ave"]
                    << std::setw(10) << it.second["min"]
                    << std::setw(10) << it.second["max"];
            }

            std::cout << "\n\n- - - - - - - - - - - - QUEUE INFORMATION - - - - - - - - - - - -"
                << std::endl << std::left << std::setw(30) << "Queue Name"
                << std::setw(10) << "Limit"
                << std::setw(10) << "Ave"
                << std::setw(10) << "Max";

            for (auto &it : queue_info) {
                std::cout << std::endl << std::left << std::setw(30) << it.first
                    << std::setw(10) << it.second["limit"]
                    << std::setw(10) << it.second["ave"]
                    << std::setw(10) << it.second["max"];
            }

            std::cout << "\n\n- - - - - - - - - - - THROUGHPUT INFORMATION - - - - - - - - - - -"
                << std::endl << std::left << std::setw(10) << "Node"
                << std::setw(10) << "Stream"
                << std::setw(10) << "Packets/second";

            for (auto &node : throughput) {
                for (auto &stream : throughput[node.first]) {
                    auto stream_rate = stream.second.second * 1000000 / stream.second.first;
                    std::cout << std::endl << std::left << std::setw(30) << node.first
                        << std::setw(10) << stream.first
                        << std::setw(10) << stream_rate;
                }
            }

            std::cout << "\n\n- - - - - - - - - - - - TRACE INFORMATION - - - - - - - - - - - -";
        }

        // Sort the JSON list
        using json_it_type = decltype(*flog.begin());
        std::sort(flog.begin(), flog.end(), [](const json_it_type a, const json_it_type b) {
            return a["ts"] < b["ts"];
        });

        // Output Trace statistics
        if (include_info) {
            bmf_nlohmann::json logstats;
            logstats["ts"] = get_duration();
            logstats["name"] = "Trace Log";
            logstats["pid"] = process_name_;
            logstats["tid"] = thread_name_;
            logstats["cat"] = categories[2];
            logstats["ph"] = phases[0];
            logstats["args"] = bmf_nlohmann::json::object();
            logstats["args"]["buffer_stats"] = bmf_nlohmann::json::array();
            logstats["args"]["buffer_size"] = trace_get_buffer_size();
            logstats["args"]["buffer_count"] = queue_map_.size();

            int buffer_allocated = 0;

            for (int i = 0; i < queue_map_.size(); i++) {
                bmf_nlohmann::json queue_info;
                queue_info["tid"] = queue_map_[i].thread_name;
                queue_info["overflowed_events"] = queue_map_[i].overflowed();
                queue_info["total_events"] = queue_map_[i].total_count();
                queue_map_[i].reset_count();
                queue_info["buffer_size"] = queue_map_[i].get_buffer_size();
                logstats["args"]["buffer_stats"].push_back(queue_info);

                // Print individual buffer info
                if (enable_printing) {
                    if (queue_info["total_events"] > 0) {
                        std::cout
                            << "\nThread: " << queue_info["tid"]
                            << "\tTotal Events: " << queue_info["total_events"]
                            << " (" << queue_info["overflowed_events"] << " overflowed)";
                        buffer_allocated++;
                    }
                }
            }

            // Print trace info
            if (enable_printing) {
                std::cout
                    << "\n\nTotal number of thread buffers allocated: "
                    << buffer_allocated << " / " << logstats["args"]["buffer_count"]
                    << "\nAllocated buffer size (individual): " << logstats["args"]["buffer_size"];
            }

            flog.push_back(logstats);
        }

        // Write formatted log file
        char tracelog_name[TRACELOG_NAME_SIZE];
        time_t now = time(0);
        strftime(tracelog_name, sizeof(tracelog_name), TRACELOG_NAME_FORMAT, localtime(&now));
        std::ofstream flog_file(tracelog_name);
        flog_file << std::setw(4) << flog << std::endl;
        flog_file.close();

        if (enable_printing) {
            auto log_time = get_duration() - final_time;
            std::cout
                << "\nTracelog formatting duration: " << log_time << " us"
                << "\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n\n";
        }
    }


#endif

END_BMF_SDK_NS
