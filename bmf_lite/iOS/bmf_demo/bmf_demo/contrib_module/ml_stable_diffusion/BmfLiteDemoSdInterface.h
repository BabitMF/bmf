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

#ifndef _BMFLITE_CONTRIB_ALFORITHM_SD_INTERFACE_H_
#define _BMFLITE_CONTRIB_ALFORITHM_SD_INTERFACE_H_

#import "BmfliteDemoMacro.h"
#import "MlStableDiffusionOC.h"
#include "bmf_lite.h"
#import <AVFoundation/AVFoundation.h>
#import <Foundation/Foundation.h>

BMFLITE_DEMO_NAMESPACE_BEGIN

class BMFLiteDemoSdInterface : public IAlgorithmInterface {
  public:
    enum Status {
        DOWNLOADING = 0,
        UNCOMPRESSING = 1,
        LOADING = 2,
        COMPLETED = 3,
        PROCESSING = 4,
        FAILED = 5,
        NO_INIT = 6,
    };

    enum RequestMode {
        NONE = 0,
        INIT_STATUS = 1,
        PROCESS_STATUS = 2,
    };

    virtual ~BMFLiteDemoSdInterface();

    virtual int setParam(Param param) override;

    virtual int processVideoFrame(VideoFrame videoframe, Param param);

    virtual int getVideoFrameOutput(VideoFrame &frame, Param &param);

    virtual int processMultiVideoFrame(std::vector<VideoFrame> videoframes,
                                       Param param);

    virtual int getMultiVideoFrameOutput(std::vector<VideoFrame> &videoframes,
                                         Param &param);

    virtual int getProcessProperty(Param &param) override;

    virtual int setInputProperty(Param attr) override;

    virtual int getOutputProperty(Param &attr) override;

  private:
    MlStableDiffusionOC *instance_ = nil;
    int reques_mode_ = 0;

    Status status_ = Status::NO_INIT;

    CGImageRef current_image_ = nil;

    uint32_t seed_ = 0;
    ;
};

BMFLITE_MODULE_CREATOR(BMFLiteDemoSdInterface)

BMFLITE_DEMO_NAMESPACE_END

#endif /* _BMFLITE_CONTRIB_ALFORITHM_SD_INTERFACE_H_ */
