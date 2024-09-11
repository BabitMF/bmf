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

#ifndef _BMFLITE_DEMO_CANNY_MODULE_H_
#define _BMFLITE_DEMO_CANNY_MODULE_H_

#import "BmfLiteDemoMacro.h"
#import "BmfLiteDemoVideoFrame.h"
#import "BmfLiteDemoModule.h"
#include "bmf_lite.h"
#include <vector>

BMFLITE_DEMO_NAMESPACE_BEGIN

class BmfLiteDemoCannyModule : public Module {
  public:
    BmfLiteDemoCannyModule();
    int init() override;
    int process(std::shared_ptr<VideoFrame> data) override;

  private:
    bmf_lite::IAlgorithm *canny_ = nullptr;
    std::vector<float> color_transform_ = {0.299f, 0.587f, 0.114f};
    float sigma_;
    float low_threshold_ = 0.2f;
    float high_threshold_ = 0.4f;
}; // class BmfLiteDemoCannyModule

BMFLITE_DEMO_NAMESPACE_END

#endif /* _BMFLITE_DEMO_CANNY_MODULE_H_ */
