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

#ifndef BMF_SDK_CBYTES_H
#define BMF_SDK_CBYTES_H
#ifdef __cplusplus

#include <memory>
#include <cstdint>
#include <cstddef>
#include "common.h"

#else
#include "stdint.h"
#include "stddef.h"
#endif

#ifdef __cplusplus
BEGIN_BMF_SDK_NS
struct BMF_API CBytes {
    uint8_t const *buffer;
    size_t size;

    //
    std::shared_ptr<uint8_t> holder;

    static CBytes make(size_t size) {
        CBytes cb;
        cb.holder = std::shared_ptr<uint8_t>(
            new uint8_t[size], [](uint8_t *ptr) { delete[] ptr; });
        cb.buffer = cb.holder.get();
        cb.size = size;
        return cb;
    }

    static CBytes make(const uint8_t *buffer, size_t size) {
        CBytes cb;
        cb.buffer = buffer;
        cb.size = size;
        return cb;
    }
};
END_BMF_SDK_NS
#endif

#endif // BMF_SDK_CBYTES_H