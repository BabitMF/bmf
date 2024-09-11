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

#include "multi_metal_texture_video_buffer.h"
#import <AVFoundation/AVFoundation.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

namespace bmf_lite {

class MultiMetalTextureVideoBufferImpl {
public:
  struct VideoTextureList *textures_;
};

MultiMetalTextureVideoBuffer::MultiMetalTextureVideoBuffer(
    void *texture, int width, int height, HardwareDataInfo hardware_data_info,
    std::shared_ptr<HWDeviceContext> device_context) {
  impl_ = std::make_shared<MultiMetalTextureVideoBufferImpl>();
  // struct VideoTextureList *texture_list = (struct VideoTextureList *)texture;
  // impl_->textures_ = new struct VideoTextureList();

  // impl_->textures_->texture_list =
  //     (VideoBuffer **)malloc(texture_list->num * sizeof(VideoBuffer *));
  // for (int i = 0; i < texture_list->num; i++) {
  //   impl_->textures_->texture_list[i] = texture_list->texture_list[i];
  // }
  impl_->textures_ = (struct VideoTextureList *)texture;
  width_ = width;
  height_ = height;
  device_context_ = device_context;
  hardware_data_info_ = hardware_data_info;
};

MultiMetalTextureVideoBuffer::~MultiMetalTextureVideoBuffer() {
  if (deleter_) {
    deleter_(this);
  }
  if (impl_) {
    for (int i = 0; i < impl_->textures_->num; i++) {
      delete impl_->textures_->texture_list[i];
      impl_->textures_->texture_list[i] = nullptr;
    }
    if (impl_->textures_->texture_list) {
      free(impl_->textures_->texture_list);
      impl_->textures_->texture_list = nullptr;
    }

    delete impl_->textures_;
    impl_->textures_ = nullptr;
  }
}

int MultiMetalTextureVideoBuffer::width() { return width_; }

int MultiMetalTextureVideoBuffer::height() { return height_; }

void *MultiMetalTextureVideoBuffer::data() { return impl_->textures_; }

std::shared_ptr<HWDeviceContext> MultiMetalTextureVideoBuffer::getHWDeviceContext() {
  return device_context_;
}

HardwareDataInfo MultiMetalTextureVideoBuffer::hardwareDataInfo() {
  return hardware_data_info_;
}

}

#endif