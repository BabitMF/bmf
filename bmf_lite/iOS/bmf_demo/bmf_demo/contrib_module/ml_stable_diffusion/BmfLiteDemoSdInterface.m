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

#include "BmfLiteDemoSdInterface.h"
#include "BmfLiteDemoLog.h"
#import <AVFoundation/AVFoundation.h>
#import <UIKit/UIKit.h>
#include <random>

BMFLITE_DEMO_NAMESPACE_BEGIN

BMFLiteDemoSdInterface::~BMFLiteDemoSdInterface() { instance_ = nil; }

int BMFLiteDemoSdInterface::setParam(Param param) {
  instance_ = [[MlStableDiffusionOC alloc] init];
  [instance_ loadAndInit];
  return 0;
}

int BMFLiteDemoSdInterface::processVideoFrame(VideoFrame videoframe,
                                              Param param) {
  std::string positive_prompt;
  if (param.getString("positive_prompt", positive_prompt) != 0) {
    return BMF_LITE_StsBadArg;
  }
  double step = 25.0f;
  if (param.getDouble("step", step) != 0) {
    step = 25.0f;
    BMFLITE_DEMO_LOGW("BMFLiteDemoSdInterface",
                      "not set step use default value 25.");
  }
  uint32_t seed;
//  if (param.getInt("seed", seed) == 0) {
//    seed_ = (uint32_t)seed;
//  } else {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> distribution(
        0, std::numeric_limits<uint32_t>::max());
    seed = distribution(gen);
      
//  }

  NSString *ns_prompt =
      [[NSString alloc] initWithCString:positive_prompt.c_str()];
  [instance_ generateImageWithPrompt:ns_prompt WithStep:step AndSeed:seed];
  return 0;
}

int BMFLiteDemoSdInterface::getVideoFrameOutput(VideoFrame &frame,
                                                Param &param) {
  if (param.getInt("request_mode", reques_mode_) != 0) {
    return BMF_LITE_StsBadArg;
  }
  if (reques_mode_ == RequestMode::INIT_STATUS) {
    int status = [instance_ getStatus];
    param.setInt("init_status", status);
    double progress_value = [instance_ getProgressValue];
    param.setDouble("progress_value", progress_value);
    return 0;
  } else if (reques_mode_ == RequestMode::PROCESS_STATUS) {
    bool completed = [instance_ hasCompleted];
    param.setInt("process_status", (int)completed);
    if (completed) {
      current_image_ = [instance_ getResult];
      bmf_lite::HardwareDataInfo hardware_info;
      hardware_info.internal_format = bmf_lite::BMF_LITE_CGImage_NONE;
      hardware_info.mem_type = bmf_lite::MemoryType::kRaw;
      bmf_lite::HardwareDeviceCreateInfo create_info{bmf_lite::kHWDeviceTypeMTL,
                                                     NULL};
      std::shared_ptr<bmf_lite::HWDeviceContext> mtl_device_context;
      bmf_lite::HWDeviceContextManager::createHwDeviceContext(
          &create_info, mtl_device_context);
      std::shared_ptr<bmf_lite::VideoBuffer> video_buffer;
      bmf_lite::VideoBufferManager::createTextureVideoBufferFromExistingData(
          current_image_, 0, 0, &hardware_info, mtl_device_context, nullptr,
          video_buffer);
      frame = VideoFrame(video_buffer);
    }
    return 0;
  }
  return -1;
}

int BMFLiteDemoSdInterface::processMultiVideoFrame(
    std::vector<VideoFrame> videoframes, Param param) {
  return -1;
}

int BMFLiteDemoSdInterface::getMultiVideoFrameOutput(
    std::vector<VideoFrame> &videoframes, Param &param) {
  return -1;
}

int BMFLiteDemoSdInterface::getProcessProperty(Param &param) { return -1; }

int BMFLiteDemoSdInterface::setInputProperty(Param attr) { return -1; }

int BMFLiteDemoSdInterface::getOutputProperty(Param &attr) { return -1; }

BMFLITE_DEMO_NAMESPACE_END
