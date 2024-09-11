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

#include "algorithm_interface.h"
#include "algorithm/bmf_contrib_algorithm.h"
#include "contrib_modules/canny/canny_algorithm.h"
#include "modules/denoise/denoise_algorithm.h"
#include "modules/super_resolution/super_resolution_algorithm.h"
#include "contrib_modules/QNNControlNet/QnnControlNet_algorithm.h"
namespace bmf_lite {

std::shared_ptr<IAlgorithmInterface>
AlgorithmInstance::createAlgorithmInterface(int algorithm_type) {
    if (algorithm_type == AlgorithmType::BMF_LITE_ALGORITHM_SUPER_RESOLUTION) {
#ifdef BMF_LITE_ENABLE_SUPER_RESOLUTION
        std::shared_ptr<SuperResolutionAlgorithm> instance =
            std::make_shared<SuperResolutionAlgorithm>();
        return instance;
#endif
    }

    if (algorithm_type == AlgorithmType::BMF_LITE_ALGORITHM_DENOISE) {
#ifdef BMF_LITE_ENABLE_DENOISE
        std::shared_ptr<DenoiseAlgorithm> instance =
            std::make_shared<DenoiseAlgorithm>();
        return instance;
#endif
    }

    if (algorithm_type == AlgorithmType::BMF_LITE_ALGORITHM_CANNY) {
#ifdef BMF_LITE_ENABLE_CANNY
        std::shared_ptr<CannyAlgorithm> instance =
            std::make_shared<CannyAlgorithm>();
        return instance;
#endif
    }
    if (algorithm_type == AlgorithmType::BMF_LITE_ALGORITHM_TEX2PIC) {
#ifdef BMF_LITE_ENABLE_TEX_GEN_PIC
        std::shared_ptr<QNNControlNetAlgorithm> instance =
            std::make_shared<QNNControlNetAlgorithm>();
        return instance;
#endif
    }

    IAlgorithmInterface *contrib_algo =
        ContribAlgorithmFactory::instance().createAlgorithmInstance(
            algorithm_type);
    if (contrib_algo != nullptr) {
        std::shared_ptr<IAlgorithmInterface> instance(contrib_algo);
        return instance;
    }
    return nullptr;
}
} // namespace bmf_lite