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

#ifndef _BMF_CANNY_ALGORITHM_H_
#define _BMF_CANNY_ALGORITHM_H_
#include "algorithm/algorithm_interface.h"
#include "algorithm/bmf_algorithm.h"
#include "media/video_buffer/video_buffer_pool.h"

namespace bmf_lite {

class CannyImpl;
class CannyAlgorithm : public IAlgorithmInterface {
  public:
    CannyAlgorithm();
    virtual ~CannyAlgorithm();

    int setParam(Param param) override;
    int unInit();

    int processVideoFrame(VideoFrame frame, Param param) override;
    int getVideoFrameOutput(VideoFrame &frame, Param &param) override;

    int processMultiVideoFrame(std::vector<VideoFrame> videoframes,
                               Param param) override;
    int getMultiVideoFrameOutput(std::vector<VideoFrame> &videoframes,
                                 Param &param) override;

    int getProcessProperty(Param &param) override;
    int setInputProperty(Param attr) override;
    int getOutputProperty(Param &attr) override;

    std::shared_ptr<CannyImpl> impl_ = nullptr;
};

} // namespace bmf_lite

#endif