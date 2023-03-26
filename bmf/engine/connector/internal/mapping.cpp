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

#include "mapping.h"

namespace bmf::internal {
    InstanceMapping<bmf_engine::Graph> &ConnectorMapping::GraphInstanceMapping() {
        static auto *graph_instance_mapping = new InstanceMapping<bmf_engine::Graph>();
        return *graph_instance_mapping;
    }

    InstanceMapping<bmf_sdk::Module> &ConnectorMapping::ModuleInstanceMapping() {
        static auto *module_instance_mapping = new InstanceMapping<bmf_sdk::Module>();
        return *module_instance_mapping;
    }

    InstanceMapping<std::function<bmf_sdk::CBytes(bmf_sdk::CBytes)> > &
    ConnectorMapping::ModuleCallbackInstanceMapping() {
        static auto *module_callback_instance_mapping =
                new InstanceMapping<std::function<bmf_sdk::CBytes(bmf_sdk::CBytes)> >();
        return *module_callback_instance_mapping;
    }

}