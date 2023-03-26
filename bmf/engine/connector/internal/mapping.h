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

#ifndef CONNECTOR_MAPPING_H
#define CONNECTOR_MAPPING_H

#include "instance_mapping.hpp"

#include "../../c_engine/include/graph.h"

#include <bmf/sdk/cbytes.h>
#include <bmf/sdk/module.h>

namespace bmf {
    namespace internal {
        class ConnectorMapping {
        public:
            static InstanceMapping<bmf_engine::Graph> &GraphInstanceMapping();

            static InstanceMapping<bmf_sdk::Module> &ModuleInstanceMapping();

            static InstanceMapping<std::function<bmf_sdk::CBytes(bmf_sdk::CBytes)> > &ModuleCallbackInstanceMapping();
        };
    }
}

#endif //CONNECTOR_MAPPING_H
