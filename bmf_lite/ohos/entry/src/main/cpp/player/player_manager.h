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
#ifndef BMFLITE_PLAYER_MANAGER_H
#define BMFLITE_PLAYER_MANAGER_H

#include <multimedia/player_framework/native_averrors.h>
#include <string>
#include <mutex>

#include "hilog/log.h"
#include <multimedia/player_framework/avplayer.h>

namespace bmf_lite_demo {

class NDKPlayer {
  public:
    NDKPlayer(OHNativeWindow *window);
    ~NDKPlayer();

    //     static void Destroy() {
    //         if (ndkPlayer_ != nullptr) {
    //             delete ndkPlayer_;
    //             ndkPlayer_ = nullptr;
    //         }
    //     }

    OH_AVErrCode SetFdSource(int32_t fd, int64_t offset, int64_t size);
    OH_AVErrCode Play();
    OH_AVErrCode Stop();
    OH_AVErrCode ReleasePlayer();

  private:
    OH_AVPlayer *player_;

    OH_AVErrCode ret_;

    OHNativeWindow *window_;
    AVPlayerCallback playerCallback_;

    static NDKPlayer *ndkPlayer_;
    static std::mutex mtx_;
    volatile bool valid_;
};

} // namespace bmf_lite_demo

#endif // BMFLITE_PLAYER_MANAGER_H
