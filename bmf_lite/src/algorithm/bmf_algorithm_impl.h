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

#ifndef _BMFLITE_ALFORITHM_IMPL_H_
#define _BMFLITE_ALFORITHM_IMPL_H_

#include "algorithm/bmf_video_frame.h"
#include "algorithm_interface.h"
#include "common/bmf_param.h"
#include "common/error_code.h"

namespace bmf_lite {

enum ModsMode {
    UPDATE_MODE = 1,
    CREATE_MODE = 2,
    DELETE_MODE = 3,
    READ_MODE = 4,
    SET_PROCESS_MODE = 5,
    CREATE_AND_PROCESS_MODE = 6,
};

class ModsParam {
  public:
    int change_mode;
    std::string instance_id;
    int algorithm_type = 0;
    int algorithm_version = 0;
};

class ModsInstance {
  public:
    std::shared_ptr<IAlgorithmInterface> instance_;
    std::string instance_id_;
};

class AlgorithmImpl {
  public:
    int setParam(Param param);
    int parseParam(Param param);

    int processVideoFrame(VideoFrame videoframe, Param param);
    int getVideoFrameOutput(VideoFrame &videoframe, Param &param);

    int processMultiVideoFrame(std::vector<VideoFrame> videoframes,
                               Param param) {
        return BMF_LITE_StsFuncNotImpl;
    }

    int getMultiVideoFrameOutput(std::vector<VideoFrame> &videoframes,
                                 Param &param) {
        return BMF_LITE_StsFuncNotImpl;
    }

    int getProcessProperty(Param &param) { return -1; }

    int setInputProperty(Param attr) { return -1; }

    int getOutputProperty(Param &attr) { return -1; }

  private:
    std::vector<ModsInstance> instances_;
    std::shared_ptr<IAlgorithmInterface> default_instance_;
    ModsParam mods_param_;
};

} // namespace bmf_lite

#endif // _BMFLITE_ALFORITHM_IMPL_H_