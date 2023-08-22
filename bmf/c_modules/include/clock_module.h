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
#ifndef BMF_OPUS_DECODER_MODULE_H
#define BMF_OPUS_DECODER_MODULE_H

#include <bmf/sdk/common.h>
#include <bmf/sdk/packet.h>
#include <bmf/sdk/task.h>
#include <bmf/sdk/module.h>
#include <bmf/sdk/module_registry.h>

#include "fraction.hpp"

#include <chrono>

USE_BMF_SDK_NS

class ClockModule : public Module {
  public:
    ClockModule(int node_id, JsonParam option);

    ~ClockModule() {}

    int process(Task &task);

    int reset() { return 0; };

    bool is_hungry(int input_stream_id);

  private:
    Fraction::Fraction fps_tick_, time_base_;
    uint64_t frm_cnt_;

    std::chrono::high_resolution_clock::duration tick_;
    std::chrono::time_point<std::chrono::high_resolution_clock> lst_ts_;
};

#endif
