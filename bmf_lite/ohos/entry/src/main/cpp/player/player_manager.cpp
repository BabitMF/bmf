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
#include "player_manager.h"
#include "common.h"
#include <unistd.h>

namespace bmf_lite_demo {

void OnPlayerInfo(OH_AVPlayer *player, AVPlayerOnInfoType type, int32_t extra) {
    //     OH_LOG_Print(LOG_APP, LOG_INFO, LOG_PRINT_DOMAIN, "NDKPlayer",
    //     "AVPlayerOnInfoType %{public}d, extra %{public}d.", type, extra);
}

void OnPlayerError(OH_AVPlayer *player, int32_t errorCode,
                   const char *errorMsg) {
    OH_LOG_Print(LOG_APP, LOG_ERROR, LOG_PRINT_DOMAIN, "NDKPlayer",
                 "OnPlayerError errorCode %{public}d, extra %{public}s.",
                 errorCode, errorMsg);
}

NDKPlayer::NDKPlayer(OHNativeWindow *window) {
    valid_ = false;
    ReleasePlayer();
    player_ = OH_AVPlayer_Create();
    if (player_ == nullptr) {
        OH_LOG_ERROR(LOG_APP, "OH_AVPlayer_Create failed.");
        return;
    }
    window_ = window;
    playerCallback_.onInfo = OnPlayerInfo;
    playerCallback_.onError = OnPlayerError;
    ret_ = OH_AVPlayer_SetPlayerCallback(player_, playerCallback_);
    if (ret_ == OH_AVErrCode::AV_ERR_OK) {
        valid_ = true;
    }
}

NDKPlayer::~NDKPlayer() {
    if (valid_ && player_ != nullptr) {
        ret_ = ReleasePlayer();
        if (ret_ == OH_AVErrCode::AV_ERR_OK) {
            OH_LOG_INFO(LOG_APP, "OH_AVPlayer_Release success.");
        } else {
            OH_LOG_ERROR(LOG_APP, "OH_AVPlayer_Release failed, %d", ret_);
        }
    }
    valid_ = false;
}

OH_AVErrCode NDKPlayer::SetFdSource(int32_t fd, int64_t offset, int64_t size) {
    if (valid_ && player_ != nullptr) {
        ret_ = OH_AVPlayer_SetFDSource(player_, fd, offset, size);
        if (ret_ == OH_AVErrCode::AV_ERR_OK) {
            OH_LOG_INFO(LOG_APP, "OH_AVPlayer_SetFDSource success.");
        } else {
            OH_LOG_ERROR(LOG_APP, "OH_AVPlayer_SetFDSource failed, %d", ret_);
        }
        ret_ = OH_AVPlayer_SetVideoSurface(player_, window_);
        if (ret_ == OH_AVErrCode::AV_ERR_OK) {
            OH_LOG_INFO(LOG_APP, "OH_AVPlayer_SetVideoSurface success.");
        } else {
            OH_LOG_ERROR(LOG_APP,
                         "OH_AVPlayer_SetVideoSurface failed, %{public}d",
                         ret_);
        }
        ret_ = OH_AVPlayer_Prepare(player_);
        if (ret_ == OH_AVErrCode::AV_ERR_OK) {
            OH_LOG_INFO(LOG_APP, "OH_AVPlayer_Prepare success.");
        } else {
            OH_LOG_ERROR(LOG_APP, "OH_AVPlayer_Prepare failed, %d", ret_);
        }
        OH_AVPlayer_SetLooping(player_, true);
        ret_ = OH_AVPlayer_Play(player_);
        if (ret_ == OH_AVErrCode::AV_ERR_OK) {
            OH_LOG_INFO(LOG_APP, "OH_AVPlayer_Play success.");
        } else {
            OH_LOG_ERROR(LOG_APP, "OH_AVPlayer_Play failed, %d", ret_);
        }
    }
    return ret_;
}

OH_AVErrCode NDKPlayer::Play() {
    if (valid_ && player_ != nullptr) {
        ret_ = OH_AVPlayer_Play(player_);
        if (ret_ == OH_AVErrCode::AV_ERR_OK) {
            OH_LOG_INFO(LOG_APP, "OH_AVPlayer_Start success.");
        } else {
            OH_LOG_ERROR(LOG_APP, "OH_AVPlayer_Start failed, %d", ret_);
        }
    }
    return ret_;
}

OH_AVErrCode NDKPlayer::Stop() {
    if (valid_ && player_ != nullptr) {
        ret_ = OH_AVPlayer_Stop(player_);
        if (ret_ == OH_AVErrCode::AV_ERR_OK) {
            OH_LOG_INFO(LOG_APP, "OH_AVPlayer_Stop success.");
        } else {
            OH_LOG_ERROR(LOG_APP, "OH_AVPlayer_Stop failed, %d", ret_);
        }
    }
    return ret_;
}

OH_AVErrCode NDKPlayer::ReleasePlayer() {
    if (valid_ && player_ != nullptr) {
        ret_ = OH_AVPlayer_Release(player_);
        if (ret_ == OH_AVErrCode::AV_ERR_OK) {
            OH_LOG_INFO(LOG_APP, "OH_AVPlayer_Release success.");
        } else {
            OH_LOG_ERROR(LOG_APP, "OH_AVPlayer_Release failed, %{public}d",
                         ret_);
        }
    }
    return ret_;
}

} // namespace bmf_lite_demo