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

#include <bmf/sdk/config.h>

#ifndef BMF_COMMON_H
#define BMF_COMMON_H

#define BEGIN_BMF_SDK_NS namespace bmf_sdk {
#define END_BMF_SDK_NS }
#define USE_BMF_SDK_NS using namespace bmf_sdk;

#define BEGIN_BMF_ENGINE_NS namespace bmf_engine {
#define END_BMF_ENGINE_NS }
#define USE_BMF_ENGINE_NS using namespace bmf_engine;

#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

#define BMF_SDK_VERSION BMF_BUILD_VERSION

// interface export
#ifdef _WIN32
#ifdef BMF_BUILD_SHARED
#define BMF_API __declspec(dllexport)
#else // BMF_BUILD_SHARED
#define BMF_API __declspec(dllimport)
#endif
#else // GNUC
#ifdef BMF_BUILD_SHARED
#define BMF_API __attribute__((__visibility__("default")))
#else // BMF_BUILD_SHARED
#define BMF_API
#endif //
#endif

#ifdef __cplusplus
BEGIN_BMF_SDK_NS
enum BmfResult { UNKNOWN_ERROR = -1, SUCCESS = 0 };

enum class BmfMode {
    NORMAL_MODE = 0,
    SERVER_MODE = 1,
    GENERATOR_MODE = 2,
    SUBGRAPH_MODE = 3,
    PUSHDATA_MODE = 4
};
END_BMF_SDK_NS
#endif
#endif // BMF_COMMON_H
