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
#ifndef CPP_TEST_HELPER_H
#define CPP_TEST_HELPER_H

#include <sstream>
#include <iostream>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <openssl/md5.h>

#include "gtest/gtest.h"
#include "nlohmann/json.hpp"

#define BMF_CPP_FILE_CHECK(output_file, expected_results)                      \
    EXPECT_EQ(true, MediaInfo(output_file).MediaCompareEquals(expected_results))

#define BMF_CPP_FILE_CHECK_MD5(output_file, md5)                               \
    EXPECT_EQ(true, MediaInfo(output_file).MediaCompareMD5(md5))

class MediaInfo {
  public:
    MediaInfo() = delete;
    MediaInfo(std::string filepath);
    bool MediaCompareEquals(std::string expected);
    bool MediaCompareMD5(const std::string &md5);

  private:
    nlohmann::json mediaJson;
    std::string filePath;
};

inline void deleteMediaFile(std::string filepath) {
    try {
        std::filesystem::remove(filepath);
    } catch (const std::filesystem::filesystem_error &err) {
        std::cout << "C++ test file delete error: " << err.what() << std::endl;
    }
}

#define BMF_CPP_FILE_REMOVE(output_file) deleteMediaFile(output_file)

#endif // CPP_TEST_HELPER_H
