#ifndef CPP_TEST_HELPER_H
#define CPP_TEST_HELPER_H

#include <sstream>
#include <iostream>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <openssl/md5.h>

#include "gtest/gtest.h"
#include "bmf_nlohmann/json.hpp"

#define BMF_CPP_FILE_CHECK(output_file, expected_results) \
    EXPECT_EQ(true, MediaInfo(output_file).MediaCompareEquals(expected_results))

#define BMF_CPP_FILE_CHECK_MD5(output_file, md5) \
    EXPECT_EQ(true, MediaInfo(output_file).MediaCompareMD5(md5))

class MediaInfo {
public:
    MediaInfo() = delete;
    MediaInfo(std::string filepath);
    bool MediaCompareEquals(std::string expected);
    bool MediaCompareMD5(const std::string& md5);
private:
    bmf_nlohmann::json mediaJson;
    std::string filePath;
};

inline void deleteMediaFile(std::string filepath) {
    try {
        std::filesystem::remove(filepath);
    } catch(const std::filesystem::filesystem_error& err) {
        std::cout << "C++ test file delete error: " << err.what() << std::endl;
    }
}

#define BMF_CPP_FILE_REMOVE(output_file) deleteMediaFile(output_file)

#endif //CPP_TEST_HELPER_H
