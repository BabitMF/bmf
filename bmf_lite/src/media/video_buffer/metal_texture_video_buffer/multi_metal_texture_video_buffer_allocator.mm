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

#include "multi_metal_texture_video_buffer_allocator.h"
#include "common/error_code.h"
#include "metal_texture_video_buffer_allocator.h"
#include "multi_metal_texture_video_buffer.h"
#include "utils/log.h"
#import <AVFoundation/AVFoundation.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

namespace bmf_lite {

MultiMetalTextureVideoBufferAllocator::MultiMetalTextureVideoBufferAllocator(){}

MultiMetalTextureVideoBufferAllocator::~MultiMetalTextureVideoBufferAllocator(){}

int MultiMetalTextureVideoBufferAllocator::allocVideoBuffer(
    int width, int height, HardwareDataInfo *data_info,
    std::shared_ptr<HWDeviceContext> device_context,
    VideoBuffer *&video_buffer) {
  void *info;
  device_context->getContextInfo(info);
  id<MTLDevice> device = (__bridge_transfer id<MTLDevice>)info;
  struct VideoTextureList *texture_list = new VideoTextureList();
  if (data_info->internal_format == BMF_LITE_MTL_NV12) {
    MetalTextureVideoBufferAllocator metal_texture_video_buffer_allocator;
    texture_list->num = 2;
    texture_list->texture_list =
        (VideoBuffer **)malloc(sizeof(VideoBuffer *) * texture_list->num);

    VideoBuffer *y_metal_texture_video_buffer = NULL;
    HardwareDataInfo temp_data_info;
    temp_data_info.internal_format = BMF_LITE_MTL_R8Unorm;
    metal_texture_video_buffer_allocator.allocVideoBuffer(
        width, height, &temp_data_info, device_context,
        y_metal_texture_video_buffer);
    y_metal_texture_video_buffer->setDeleter([](VideoBuffer *video_buffer) {
      MetalTextureVideoBufferAllocator::releaseVideoBuffer(video_buffer);
    });

    texture_list->texture_list[0] = y_metal_texture_video_buffer;

    VideoBuffer *uv_metal_texture_video_buffer = NULL;
    temp_data_info.internal_format = BMF_LITE_MTL_RG8Unorm;
    metal_texture_video_buffer_allocator.allocVideoBuffer(
        width, height, &temp_data_info, device_context,
        uv_metal_texture_video_buffer);
    y_metal_texture_video_buffer->setDeleter([](VideoBuffer *video_buffer) {
      MetalTextureVideoBufferAllocator::releaseVideoBuffer(video_buffer);
    });
    texture_list->texture_list[1] = uv_metal_texture_video_buffer;
  }

  video_buffer = new MultiMetalTextureVideoBuffer(texture_list, width, height,
                                                  *data_info, device_context);
  video_buffer->setDeleter([](VideoBuffer *video_buffer) {
    MultiMetalTextureVideoBufferAllocator::releaseVideoBuffer(video_buffer);
  });
  return 0;
}

int MultiMetalTextureVideoBufferAllocator::releaseVideoBuffer(
    VideoBuffer *video_buffer) {
  return 0;
}

}

#endif