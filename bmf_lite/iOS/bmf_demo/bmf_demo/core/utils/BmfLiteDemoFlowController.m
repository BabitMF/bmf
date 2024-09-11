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

#import "BmfLiteDemoFlowController.h"

BMFLITE_DEMO_NAMESPACE_BEGIN

void FlowController::start() {
    start_ = std::chrono::high_resolution_clock::now();
    first_ = true;
    current_count = 0;
}

void FlowController::startInCMTime() {
    first_ = true;
    time_gap_ = 0;
}

void FlowController::controlByVideoPts(CMTime pts){
    if (first_) {
        pre_pts = pts;
        first_ = false;
        time_gap_ = 0;
        start_ = std::chrono::steady_clock::now();
        return;
    }
    double m_gap = (CMTimeGetSeconds(pts) - CMTimeGetSeconds(pre_pts)) * 1000.0;
    end_ = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> diff = end_ - start_;
    auto elapsed = m_gap - diff.count();
    if (elapsed > 0) {
        std::chrono::duration<double, std::milli> sleep_time(elapsed);
        std::this_thread::sleep_for(sleep_time);
    }
}

bool FlowController::frameIsDelay(CMTime pts) {
    if (flag) {
        pre_pts = pts;
        flag = false;
        return false;
    }
    std::chrono::time_point<std::chrono::steady_clock>  end = std::chrono::steady_clock::now();
    double m_gap = (CMTimeGetSeconds(pts) - CMTimeGetSeconds(pre_pts)) * 1000.0;
    std::chrono::duration<double, std::milli> diff = end - start_;
    return m_gap < diff.count();
}

BMFLITE_DEMO_NAMESPACE_END
