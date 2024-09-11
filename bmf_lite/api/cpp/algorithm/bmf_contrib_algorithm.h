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

#ifndef _BMFLITE_CONTRIB_ALFORITHM_INTERFACE_H_
#define _BMFLITE_CONTRIB_ALFORITHM_INTERFACE_H_

#include "algorithm/bmf_algorithm.h"
#include "algorithm/bmf_video_frame.h"
#include "common/bmf_param.h"
#include <functional>
#include <map>
#include <mutex>
#include <string>

namespace bmf_lite {

class ContribAlgorithmFactory {
  public:
    static ContribAlgorithmFactory &instance() {
        static ContribAlgorithmFactory *factory = nullptr;
        static std::once_flag flag;

        std::call_once(flag,
                       [&]() { factory = new ContribAlgorithmFactory(); });
        return *factory;
    }

    virtual ~ContribAlgorithmFactory();

    IAlgorithmInterface *createAlgorithmInstance(int algorithm_type);

    bool registerCreator(int algorithm_type,
                         std::function<IAlgorithmInterface *()> pfunc);

  private:
    ContribAlgorithmFactory();
    std::map<int, std::function<IAlgorithmInterface *()>> func_mp_;
};

#define BMFLITE_MODULE_CREATOR(NAME)                                           \
    class NAME##Creator {                                                      \
      public:                                                                  \
        static IAlgorithmInterface *createAlgorithmInstance() {                \
            return new NAME();                                                 \
        }                                                                      \
    };

#define BMFLITE_MODEL_REGIST(ALGO_TYPE, NAME)                                  \
    do {                                                                       \
        auto &factory = bmf_lite::ContribAlgorithmFactory::instance();         \
        factory.registerCreator(ALGO_TYPE,                                     \
                                &NAME##Creator::createAlgorithmInstance);      \
    } while (0);

} // namespace bmf_lite

#endif // _BMFLITE_CONTRIB_ALFORITHM_INTERFACE_H_
