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
#include <bmf/sdk/exception_factory.h>

#include <gtest/gtest.h>

USE_BMF_SDK_NS

void test_bmf_error() {
    BMF_Error(-2, "some error happen");
}

void test_bmf_error_(std::string error_str) {
    BMF_Error_(-1, "some error happen %s\n", error_str.c_str());
}

TEST(exception, simple_exception) {
    Exception e;
    try {
        test_bmf_error();
    }
    catch (Exception e) {
        std::cout << e.what() << std::endl;
    }
    try {
        std::string test_error_info = "hello_world";
        test_bmf_error_(test_error_info);
    }
    catch (Exception e) {
        std::cout << e.what() << std::endl;
    }
}