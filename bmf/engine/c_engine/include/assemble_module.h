/*
 * Copyright 2024 Babit Authors
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

#ifndef BMF_ASSEMBLE_MODULE_H
#define BMF_ASSEMBLE_MODULE_H

#include <bmf/sdk/module.h>
#include <bmf/sdk/module_registry.h>

USE_BMF_SDK_NS
class AssembleModule : public Module {
  public:
    AssembleModule(int node_id, JsonParam json_param);

    int reset();

    int process(Task &task);

    int close();

    std::map<int, bool> in_eof_;

    int last_input_num_;

    int last_output_num_;

    int queue_index_;

    std::map<int, std::shared_ptr<std::queue<Packet>>> queue_map_;
};

REGISTER_MODULE_CLASS(AssembleModule)

#endif // BMF_PASS_THROUGH_MODULE_H
