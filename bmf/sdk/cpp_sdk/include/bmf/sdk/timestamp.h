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

#ifndef BMF_TIMESTAMP_H
#define BMF_TIMESTAMP_H

#ifdef __cplusplus
#include <bmf/sdk/common.h>
BEGIN_BMF_SDK_NS
#endif
enum Timestamp : int64_t {
    UNSET = -1,
    BMF_PAUSE = 9223372036854775802,
    DYN_EOS = 9223372036854775803, // dynamical graph updated end of stream
    BMF_EOF = 9223372036854775804,
    EOS = 9223372036854775805,
    INF_SRC = 9223372036854775806,
    DONE = 9223372036854775807
};
#ifdef __cplusplus
END_BMF_SDK_NS
#endif

#endif // BMF_TIMESTAMP_H
