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

#include "canny_algorithm.h"
#include "algorithm/bmf_video_frame.h"
#include "common/error_code.h"
#include "media/video_buffer/metal_texture_video_buffer/mtl_device_context.h"
#include "media/video_buffer/transform/cvpixelbuffer_transformer.h"
#include "metal/metal_helper.h"
#include <cmath>
#import <AVFoundation/AVFoundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#if defined(__APPLE__) && defined(BMF_LITE_ENABLE_CANNY)

namespace bmf_lite {

class CannyImpl {
public:
  int parseInitParam(Param param) {
    int fmt;
    if (param.getInt("cvpixelbuffer_fmt", fmt) != 0) {
      return BMF_LITE_StsBadArg;
    }
    cvpixelbuffer_fmt_ = (uint32_t)fmt;
    if (param.getFloat("low_threshold", low_threshold_) != 0) {
      return BMF_LITE_StsBadArg;
    }
    if (param.getFloat("high_threshold", high_threshold_) != 0) {
      return BMF_LITE_StsBadArg;
    }
    if (param.getFloat("sigma", sigma_) != 0) {
      return BMF_LITE_StsBadArg;
    }
    std::vector<float> color_transform;
    if (param.getFloatList("color_transform", color_transform) != 0) {
      return BMF_LITE_StsBadArg;
    }
    color_transform_[0] = color_transform[0];
    color_transform_[1] = color_transform[1];
    color_transform_[2] = color_transform[2];
    return BMF_LITE_StsOk;
  }

  int setParam(Param param) {
    if (!@available(iOS 14.0, *)) {
      return BMF_LITE_StsNotSupport;
    }
    int ret = parseInitParam(param);
    if (ret != BMF_LITE_StsOk) {
      return ret;
    }
    @autoreleasepool {
      canny_ = [[MPSImageCanny alloc] initWithDevice:metal::MetalHelper::instance().mtl_device()
                          linearToGrayScaleTransform:&color_transform_[0]
                                              sigma:sigma_];
    }
    if (canny_ == nil) {
      return BMF_LITE_StsNotSupport;
    }
    command_queue_ = [metal::MetalHelper::instance().mtl_device() newCommandQueue];

    std::shared_ptr<MtlDeviceContext> mtl_device_context = std::make_shared<MtlDeviceContext>();
    mtl_device_context->create_context();

    HardwareDataInfo hardware_data_info_in;
    hardware_data_info_in.mem_type = MemoryType::kCVPixelBuffer;

    HardwareDataInfo hardware_data_info_out;
    hardware_data_info_out.mem_type = MemoryType::kMultiMetalTexture;

    trans_ = std::make_shared<CvPixelBufferTransformer>();
    ret = trans_->init(hardware_data_info_in, mtl_device_context, hardware_data_info_out);
    if (ret != BMF_LITE_StsOk) {
      return ret;
    }

    return BMF_LITE_StsOk;
  }

  int processVideoFrame(VideoFrame *in_frame) {
    std::shared_ptr<VideoBuffer> ibuf = in_frame->buffer();
    if (ibuf == nullptr) {
      return BMF_LITE_StsBadArg;
    }

    std::shared_ptr<VideoBuffer> temp_ibuf = nullptr;
    int ret = trans_->trans(ibuf, temp_ibuf);
    if (ret != BMF_LITE_StsOk) {
      return ret;
    }

    VideoFrame temp_iframe(temp_ibuf);

    VideoTextureList *temp_multi_data = (VideoTextureList *)(temp_iframe.buffer()->data());
    id<MTLTexture> tex =(__bridge_transfer id<MTLTexture>)(temp_multi_data->texture_list[0]->data());

    auto &helper = metal::MetalHelper::instance();
    if (last_width_ != tex.width || last_height_ != tex.height || cur_tex_ == nil) {
      cur_tex_ = nil;
      if (BMF_LITE_StsOk != helper.gen_tex(&cur_tex_, tex.pixelFormat, tex.width, tex.height,
                            MTLTextureUsageShaderWrite | MTLTextureUsageShaderRead, MTLStorageModePrivate)) {
        return BMF_LITE_MetalCreateTextureFailed;
      }
      last_width_ = tex.width;
      last_height_ = tex.height;
    }

    id<MTLCommandBuffer> command_buffer = [command_queue_ commandBuffer];
    if (command_buffer == nil) {
      return BMF_LITE_CommandBufferFailed;
    }

    [canny_ encodeToCommandBuffer:command_buffer sourceTexture:tex destinationTexture:cur_tex_];

    id<MTLBlitCommandEncoder> blit_command_encoder = [command_buffer blitCommandEncoder];
    if (@available(iOS 13.0, *)) {
      [blit_command_encoder copyFromTexture:cur_tex_ toTexture:tex];
    } else {
      [blit_command_encoder
            copyFromTexture:cur_tex_
                sourceSlice:0
                sourceLevel:0
               sourceOrigin:MTLOriginMake(0, 0, 0)
                 sourceSize:MTLSizeMake(tex.width, tex.height, tex.depth)
                  toTexture:tex
           destinationSlice:0
           destinationLevel:0
          destinationOrigin:MTLOriginMake(0, 0, 0)];
    }
    [blit_command_encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if (command_buffer.status != MTLCommandBufferStatusCompleted) {
      return BMF_LITE_MetalShaderExecFailed;
    }
    return BMF_LITE_StsOk;
  }

  CannyImpl() {}

  ~CannyImpl() {
    command_queue_ = nil;
    trans_ = nil;
    canny_ = nil;
  }

private:
  id<MTLCommandQueue> command_queue_ = nil;
  MPSImageCanny *canny_ = nil;
  std::shared_ptr<bmf_lite::CvPixelBufferTransformer> trans_ = nullptr;
  OSType cvpixelbuffer_fmt_ = kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange;
  float color_transform_[3] = {0.299f, 0.587f, 0.114f}; // bt601/JPEG
  id<MTLTexture> cur_tex_ = nil;
  int last_width_ = 0;
  int last_height_ = 0;
  float sigma_ = std::sqrt(2);
  float low_threshold_ = 0.2f;
  float high_threshold_ = 0.4f;
};

int CannyAlgorithm::setParam(Param param) {
  if (impl_ == nullptr) {
    impl_.reset(new (std::nothrow) CannyImpl());
  }
  return impl_->setParam(param);
}

int CannyAlgorithm::processVideoFrame(VideoFrame frame, Param param) {
  return impl_->processVideoFrame(&frame);
}

int CannyAlgorithm::getVideoFrameOutput(VideoFrame &frame, Param &param) {
  return BMF_LITE_StsFuncNotImpl;
}

int CannyAlgorithm::processMultiVideoFrame(std::vector<VideoFrame> videoframes,
                                           Param param) {
  return BMF_LITE_StsFuncNotImpl;
}

int CannyAlgorithm::getMultiVideoFrameOutput(
    std::vector<VideoFrame> &videoframes, Param &param) {
  return BMF_LITE_StsFuncNotImpl;
}

int CannyAlgorithm::unInit() { impl_ = nullptr; }

CannyAlgorithm::CannyAlgorithm() {}

CannyAlgorithm::~CannyAlgorithm() {}

int CannyAlgorithm::getProcessProperty(Param &param) {
  return BMF_LITE_StsFuncNotImpl;
}

int CannyAlgorithm::setInputProperty(Param attr) {
  return BMF_LITE_StsFuncNotImpl;
}

int CannyAlgorithm::getOutputProperty(Param &attr) {
  return BMF_LITE_StsFuncNotImpl;
}

}

#endif
