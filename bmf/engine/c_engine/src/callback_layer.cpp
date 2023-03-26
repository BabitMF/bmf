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

#include "../include/callback_layer.h"

#include <utility>

BEGIN_BMF_ENGINE_NS
    void ModuleCallbackLayer::add_callback(int64_t key, std::function<CBytes(CBytes)> callback) {
        callback_binding[key] = std::move(callback);
    }

    CBytes ModuleCallbackLayer::call(int64_t key, CBytes para) {
        if (!callback_binding.count(key))
            return CBytes::make(nullptr, 0);
        return callback_binding[key](para);
    }
END_BMF_ENGINE_NS