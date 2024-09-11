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

#include "BmfLiteDemoCannyModule.h"
#include "BmfLiteDemoErrorCode.h"
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <Metal/Metal.h>
#include "BmfLiteDemoMetalHelper.h"

BMFLITE_DEMO_NAMESPACE_BEGIN

BmfLiteDemoCannyModule::BmfLiteDemoCannyModule() {
    canny_ = nullptr;
}

int BmfLiteDemoCannyModule::init() {
    canny_ = bmf_lite::AlgorithmFactory::createAlgorithmInterface();
    bmf_lite::Param init_param;
    init_param.setInt("change_mode",2);
    init_param.setString("instance_id","canny1");
    init_param.setInt("algorithm_type", 2);

    init_param.setInt("algorithm_version",0);
    init_param.setFloat("sigma", sigma_);
    
    init_param.setFloat("low_threshold", low_threshold_);
    init_param.setFloat("high_threshold", high_threshold_);

    init_param.setInt("cvpixelbuffer_fmt", kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange);
    init_param.setFloatList("color_transform", color_transform_);
    assert(canny_->setParam(init_param) == 0);
    
    bmf_lite::Param set_param;
    set_param.setInt("change_mode",5);
    set_param.setString("instance_id","canny1");
    set_param.setInt("algorithm_type", 2);
    set_param.setInt("algorithm_version", 0);
    assert(canny_->setParam(set_param) == 0);
    

    HardwareDataInfo hardware_data_info_in;
    hardware_data_info_in.mem_type = MemoryType::kCVPixelBuffer;

    HardwareDataInfo hardware_data_info_out;
    hardware_data_info_out.mem_type = MemoryType::kMultiMetalTexture;

    bmf_lite::HardwareDeviceCreateInfo create_info{bmf_lite::kHWDeviceTypeMTL, NULL};
    std::shared_ptr<bmf_lite::HWDeviceContext> mtl_device_context;
    bmf_lite::HWDeviceContextManager::createHwDeviceContext(&create_info,mtl_device_context);
      
    return 0;
}

int BmfLiteDemoCannyModule::process(std::shared_ptr<VideoFrame> data) {
    if (data->eos_) {
        return 0;
    }

    @autoreleasepool {
        bmf_lite::HardwareDataInfo hardware_info{};
        hardware_info.internal_format=bmf_lite::BMF_LITE_CV_NV12;
        hardware_info.mem_type=bmf_lite::MemoryType::kCVPixelBuffer;

        bmf_lite::HardwareDeviceCreateInfo create_info{bmf_lite::kHWDeviceTypeMTL, NULL};
        std::shared_ptr<bmf_lite::HWDeviceContext> mtl_device_context;
        bmf_lite::HWDeviceContextManager::createHwDeviceContext(&create_info,mtl_device_context);
        std::shared_ptr<bmf_lite::VideoBuffer> video_buffer;
        int w = CVPixelBufferGetWidth(data->buffer_);
        int h = CVPixelBufferGetHeight(data->buffer_);
        bmf_lite::VideoBufferManager::createTextureVideoBufferFromExistingData(data->buffer_, w, h, &hardware_info, mtl_device_context, nullptr, video_buffer);
    
        bmf_lite::VideoFrame videoframe(video_buffer);
        bmf_lite::Param param;
        assert(canny_->processVideoFrame(videoframe, param) == 0);
    }
    return SUCCESS;
}

BMFLITE_DEMO_NAMESPACE_END
