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

#ifndef BMF_ENGINE_CALLBACK_LAYER_H
#define BMF_ENGINE_CALLBACK_LAYER_H

#include <bmf/sdk/cbytes.h>

#include <bmf/sdk/common.h>

#include <functional>
#include <map>
#include <vector>

USE_BMF_SDK_NS
BEGIN_BMF_ENGINE_NS
    class ModuleCallbackLayer {
    public:
        ModuleCallbackLayer() {}

        void add_callback(int64_t key, std::function<CBytes(CBytes)> callback);

        CBytes call(int64_t key, CBytes para);

    private:
        std::map<int64_t, std::function<CBytes(CBytes)> > callback_binding;
    };
END_BMF_ENGINE_NS

#endif //BMF_ENGINE_CALLBACK_LAYER_H
