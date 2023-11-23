#include "builder.hpp"
#include "nlohmann/json.hpp"

#include "cpp_test_helper.h"

static int logcallback(int level, const char *msg) {
    std::cout<< "log_func succeed" << " " << "level: " << level << "msg: " << msg << std::endl;
    return 0;
}

TEST(cpp_basefeature, log_callback) {
    BMFLOG_SET_LOG_CALLBACK(logcallback);
    BMFLOG(BMF_INFO) << "test info log: " << 5 << 'a' << std::endl;
    BMFLOG(BMF_ERROR) << "test info log: " << 5 << 'a' << std::endl;
    BMFLOG(BMF_DEBUG) << "test info log: " << 5 << 'a' << std::endl;
    BMFLOG(BMF_WARNING) << "test info log: " << 5 << 'a' << std::endl;
    BMFLOG(BMF_FATAL) << "test info log: " << 5 << 'a' << std::endl;
    BMFLOG_SET_LOG_CALLBACK(nullptr);
    BMFLOG(BMF_INFO) << "test info log" << 5 << 'a' << std::endl;
    BMFLOG(BMF_ERROR) << "test info log" << 5 << 'a' << std::endl;
    BMFLOG(BMF_DEBUG) << "test info log" << 5 << 'a' << std::endl;
    BMFLOG(BMF_WARNING) << "test info log" << 5 << 'a' << std::endl;
    BMFLOG(BMF_FATAL) << "test info log" << 5 << 'a' << std::endl;
}