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

#ifndef _BMF_LITE_ALGORITHM_ENGINE_H_
#define _BMF_LITE_ALGORITHM_ENGINE_H_

#include "bmf_video_frame.h"
#include "common/bmf_common.h"
#include "common/bmf_param.h"
#include <memory>
#include <vector>

namespace bmf_lite {
class AlgorithmImpl;
class BMF_LITE_EXPORT IAlgorithm {
  public:
    IAlgorithm();
    ~IAlgorithm();

    int setParam(Param param);

    int processVideoFrame(VideoFrame videoframe, Param param);
    int getVideoFrameOutput(VideoFrame &videoframe, Param &param);

    int processMultiVideoFrame(std::vector<VideoFrame> videoframes,
                               Param param);
    int getMultiVideoFrameOutput(std::vector<VideoFrame> &videoframes,
                                 Param &param);

    int getProcessProperty(Param &param);
    int setInputProperty(Param attr);
    int getOutputProperty(Param &attr);

  private:
    AlgorithmImpl *impl_ = nullptr;
};

class BMF_LITE_EXPORT AlgorithmFactory {
  public:
    static IAlgorithm *createAlgorithmInterface();
    static void releaseAlgorithmInterface(IAlgorithm *instance);
};

class IAlgorithmInterface {
  public:
    virtual ~IAlgorithmInterface();

    virtual int setParam(Param param);

    virtual int processVideoFrame(VideoFrame videoframe, Param param);

    virtual int getVideoFrameOutput(VideoFrame &frame, Param &param);

    virtual int processMultiVideoFrame(std::vector<VideoFrame> videoframes,
                                       Param param);

    virtual int getMultiVideoFrameOutput(std::vector<VideoFrame> &videoframes,
                                         Param &param);

    virtual int getProcessProperty(Param &param);

    virtual int setInputProperty(Param attr);

    virtual int getOutputProperty(Param &attr);
};

} // namespace bmf_lite

#endif // _BMF_LITE_ALGORITHM_ENGINE_H_