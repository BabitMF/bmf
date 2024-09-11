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

#ifndef _BMFLITE_DEMO_FLOW_CONTROLLER_H_
#define _BMFLITE_DEMO_FLOW_CONTROLLER_H_

#import "BmfLiteDemoMacro.h"
#import <Foundation/Foundation.h>
#import <AVFoundation/AVFoundation.h>
#include <chrono>
#include <thread>

BMFLITE_DEMO_NAMESPACE_BEGIN

class FlowController {
  public:
    FlowController() = default;
    FlowController(double frame_rate) : frame_rate_(frame_rate) {}
    void start();

    void startInCMTime();
    inline double getFrameRate() const { return frame_rate_; }

    inline void setFrameRate(double frame_rate) { frame_rate_ = frame_rate; }

    void controlByVideoPts(CMTime pts);

    bool frameIsDelay(CMTime pts);

  private:
    CMTime pre_pts;
    double frame_rate_{30};
    double time_gap_ = 0;
    bool first_ = true;
    double current_count = 0;
    std::chrono::time_point<std::chrono::steady_clock> start_, end_;
    bool flag = true;
}; // end class BMFModsFlowController

BMFLITE_DEMO_NAMESPACE_END

#endif /* _BMFLITE_DEMO_FLOW_CONTROLLER_H_ */
