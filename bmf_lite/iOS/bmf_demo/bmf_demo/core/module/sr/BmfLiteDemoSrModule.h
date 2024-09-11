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

#ifndef _BMFLITE_DEMO_SR_MODULE_H_
#define _BMFLITE_DEMO_SR_MODULE_H_

#import "BmfLiteDemoMacro.h"
#import "BmfLiteDemoVideoFrame.h"
#import "BmfLiteDemoModule.h"
#include "bmf_lite.h"

BMFLITE_DEMO_NAMESPACE_BEGIN

class BmfLiteDemoSrModule : public Module {
  public:
    BmfLiteDemoSrModule();
    ~BmfLiteDemoSrModule();

    int init() override;

    int process(std::shared_ptr<VideoFrame> data) override;

  private:
    bmf_lite::IAlgorithm *sr_ = nullptr;
}; // end class BmfLiteDemoSrModule

BMFLITE_DEMO_NAMESPACE_END

#endif /* _BMFLITE_DEMO_SR_MODULE_H_ */
