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

#ifndef _BMFLITE_ALFORITHM_INTERFACE_H_
#define _BMFLITE_ALFORITHM_INTERFACE_H_

#include "algorithm/bmf_algorithm.h"
#include "algorithm/bmf_video_frame.h"
#include "common/bmf_param.h"

namespace bmf_lite {

enum AlgorithmType {
    BMF_LITE_ALGORITHM_SUPER_RESOLUTION = 0,
    BMF_LITE_ALGORITHM_DENOISE = 1,
    BMF_LITE_ALGORITHM_CANNY = 2,
    BMF_LITE_ALGORITHM_TEX2PIC = 3,
    // more algorithm enumerations added here
};

class AlgorithmInstance {
  public:
    static std::shared_ptr<IAlgorithmInterface>
    createAlgorithmInterface(int algorithm_type);
};

} // namespace bmf_lite

#endif // _BMFLITE_ALFORITHM_INTERFACE_H_