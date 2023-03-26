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

#ifndef BMF_ENGINE_COMMON_H
#define BMF_ENGINE_COMMON_H

#include <bmf/sdk/config.h>

#define BEGIN_BMF_ENGINE_NS namespace bmf_engine {
#define END_BMF_ENGINE_NS }
#define USE_BMF_ENGINE_NS using namespace bmf_engine;

#define likely(x)      __builtin_expect(!!(x), 1)
#define unlikely(x)    __builtin_expect(!!(x), 0)

#define BMF_VERSION BMF_BUILD_VERSION
#define BMF_COMMIT BMF_BUILD_COMMIT

#endif //BMF_ENGINE_COMMON_H
