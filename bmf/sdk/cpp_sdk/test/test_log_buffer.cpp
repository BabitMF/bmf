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
#include <gtest/gtest.h>
#ifdef BMF_ENABLE_FFMPEG
extern "C" {
#include <libavcodec/avcodec.h>
};

USE_BMF_SDK_NS

void register_av_log_set_callback()
{
    static std::once_flag flag;
    std::call_once(flag, [](){
        LogBuffer::register_av_log_set_callback((void*)av_log_set_callback);
    });
}



TEST(log_buffer, init_by_string_vector) {
    register_av_log_set_callback();

    std::vector<std::string> buffer_str;
    LogBuffer log_buffer = LogBuffer(buffer_str);
    EXPECT_EQ(buffer_str.size(), 0);
    av_log(NULL, AV_LOG_INFO, "mock information 001");
    av_log(NULL, AV_LOG_INFO, "mock information 002");
    EXPECT_EQ(buffer_str.size(), 2);
    std::string info1 = "mock information 001";
    std::string info2 = "mock information 002";
    EXPECT_EQ(buffer_str[0], info1);
    EXPECT_EQ(buffer_str[1], info2);
}

TEST(log_buffer, init_by_function) {
    register_av_log_set_callback();

    std::vector<std::string> buffer_str;
    std::function<void(const std::string)> log_callback = [&buffer_str](std::string const log) -> void {
                                                buffer_str.push_back(log);
                                            };
    std::string log_level = "info";
    LogBuffer log_buffer = LogBuffer(log_callback, log_level);
    EXPECT_EQ(buffer_str.size(), 0);
    av_log(NULL, AV_LOG_INFO, "mock information 1");
    av_log(NULL, AV_LOG_INFO, "mock information 2");
    EXPECT_EQ(buffer_str.size(), 2);
    std::string info1 = "mock information 1";
    std::string info2 = "mock information 2";
    EXPECT_EQ(buffer_str[0], info1);
    EXPECT_EQ(buffer_str[1], info2);
}

#endif