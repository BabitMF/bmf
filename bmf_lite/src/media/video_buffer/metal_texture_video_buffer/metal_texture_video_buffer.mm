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

#ifdef BMF_LITE_ENABLE_METALBUFFER

#include "metal_texture_video_buffer.h"
#import <AVFoundation/AVFoundation.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

namespace bmf_lite {

class MetalTextureVideoBufferImpl {
public:
  id<MTLTexture> texture;
};

MetalTextureVideoBuffer::MetalTextureVideoBuffer(
    void *texture_id, int width, int height,
    HardwareDataInfo hardware_data_info,
    std::shared_ptr<HWDeviceContext> device_context) {
  impl_ = std::make_shared<MetalTextureVideoBufferImpl>();
  impl_->texture = (__bridge_transfer id<MTLTexture>)texture_id;

  width_ = width;
  height_ = height;
  device_context_ = device_context;
  hardware_data_info_ = hardware_data_info;
}

MetalTextureVideoBuffer::~MetalTextureVideoBuffer() {
  if (deleter_) {
    deleter_(this);
  }
}

int MetalTextureVideoBuffer::width() { return width_; }

int MetalTextureVideoBuffer::height() { return height_; }

void *MetalTextureVideoBuffer::data() {
  return (__bridge_retained void *)(impl_->texture);
}

std::shared_ptr<HWDeviceContext> MetalTextureVideoBuffer::getHWDeviceContext() {
  return device_context_;
}

HardwareDataInfo MetalTextureVideoBuffer::hardwareDataInfo() {
  return hardware_data_info_;
}

}

#endif