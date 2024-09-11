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

#include "bmf_algorithm_impl.h"
#include "algorithm_interface.h"
#include "common/error_code.h"

#include "algorithm/bmf_algorithm.h"
#include <iostream>
// #include "utils/log.h"
namespace bmf_lite {

int AlgorithmImpl::parseParam(Param param) {
    if (param.getInt("change_mode", mods_param_.change_mode) != 0) {
        return BMF_LITE_StsBadArg;
    }
    if (param.getString("instance_id", mods_param_.instance_id) != 0) {
        return BMF_LITE_StsBadArg;
    }
    if (param.getInt("algorithm_type", mods_param_.algorithm_type) != 0) {
        return BMF_LITE_StsBadArg;
    }
    if (param.getInt("algorithm_version", mods_param_.algorithm_version) != 0) {
        return BMF_LITE_StsBadArg;
    }
    return BMF_LITE_StsOk;
}

int AlgorithmImpl::setParam(Param param) {
    int res = parseParam(param);
    if (res < 0) {
        return res;
    }
    if (mods_param_.change_mode == ModsMode::CREATE_MODE) {
        ModsInstance mods_instance;
        mods_instance.instance_id_ = mods_param_.instance_id;
        mods_instance.instance_ = AlgorithmInstance::createAlgorithmInterface(
            mods_param_.algorithm_type);
        if (mods_instance.instance_ == NULL) {
            return BMF_LITE_StsBadArg;
        }
        res = mods_instance.instance_->setParam(param);
        if (res != BMF_LITE_StsOk) {
            return res;
        }
        instances_.push_back(mods_instance);
    }

    if (mods_param_.change_mode == ModsMode::UPDATE_MODE) {
        for (size_t i = 0; i < instances_.size(); i++) {
            if (instances_[i].instance_id_ == mods_param_.instance_id) {
                instances_[i].instance_->setParam(param);
            }
        }
    }

    if (mods_param_.change_mode == ModsMode::DELETE_MODE) {
        for (size_t i = 0; i < instances_.size(); i++) {
            if (instances_[i].instance_id_ == mods_param_.instance_id) {
                instances_.erase(instances_.begin() + i);
                break;
            }
        }
    }

    if (mods_param_.change_mode == ModsMode::SET_PROCESS_MODE) {
        for (size_t i = 0; i < instances_.size(); i++) {
            if (instances_[i].instance_id_ == mods_param_.instance_id) {
                default_instance_ = instances_[i].instance_;
                return BMF_LITE_StsOk;
            }
        }
        return BMF_LITE_StsBadArg;
    }

    if (mods_param_.change_mode == ModsMode::CREATE_AND_PROCESS_MODE) {
        ModsInstance mods_instance;
        mods_instance.instance_id_ = mods_param_.instance_id;
        mods_instance.instance_ = AlgorithmInstance::createAlgorithmInterface(
            mods_param_.algorithm_type);
        if (mods_instance.instance_ == NULL) {
            return BMF_LITE_StsBadArg;
        }

        res = mods_instance.instance_->setParam(param);
        if (res != BMF_LITE_StsOk) {
            return res;
        }
        instances_.push_back(mods_instance);
        default_instance_ = mods_instance.instance_;
    }
    return BMF_LITE_StsOk;
}

int AlgorithmImpl::processVideoFrame(VideoFrame videoframe, Param param) {
    return default_instance_->processVideoFrame(videoframe, param);
}

int AlgorithmImpl::getVideoFrameOutput(VideoFrame &videoframe, Param &param) {
    return default_instance_->getVideoFrameOutput(videoframe, param);
}

} // namespace bmf_lite