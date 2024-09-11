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

#include "algorithm/bmf_contrib_algorithm.h"

namespace bmf_lite {

ContribAlgorithmFactory::ContribAlgorithmFactory() {}

ContribAlgorithmFactory::~ContribAlgorithmFactory() {}

IAlgorithmInterface *
ContribAlgorithmFactory::createAlgorithmInstance(int algorithm_type) {
    auto it = func_mp_.find(algorithm_type);
    if (it != func_mp_.end()) {
        return it->second();
    }
    return nullptr;
}

bool ContribAlgorithmFactory::registerCreator(
    int algorithm_type, std::function<IAlgorithmInterface *()> pfunc) {
    if (pfunc == nullptr) {
        return false;
    }
    return func_mp_.insert(std::make_pair(algorithm_type, pfunc)).second;
}

} // namespace bmf_lite
