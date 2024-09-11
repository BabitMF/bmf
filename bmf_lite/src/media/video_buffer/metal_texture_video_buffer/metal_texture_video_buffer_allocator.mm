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

#include "metal_texture_video_buffer_allocator.h"
#include "common/error_code.h"
#include "metal_texture_video_buffer.h"
#include "utils/log.h"
#import <AVFoundation/AVFoundation.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

namespace bmf_lite {

MetalTextureVideoBufferAllocator::MetalTextureVideoBufferAllocator() {}

MetalTextureVideoBufferAllocator::~MetalTextureVideoBufferAllocator() {}

int MetalTextureVideoBufferAllocator::allocVideoBuffer(
    int width, int height, HardwareDataInfo *data_info,
    std::shared_ptr<HWDeviceContext> device_context,
    VideoBuffer *&video_buffer) {
  void *info;
  device_context->getContextInfo(info);
  id<MTLDevice> device = (__bridge_transfer id<MTLDevice>)info;
  MTLPixelFormat pixel_format = MTLPixelFormatR8Unorm;
  if (data_info->internal_format == BMF_LITE_MTL_R8Unorm) {
    pixel_format = MTLPixelFormatR8Unorm;
  } else if (data_info->internal_format == BMF_LITE_MTL_RG8Unorm) {
    pixel_format = MTLPixelFormatRG8Unorm;
  }
  MTLTextureDescriptor *td =
      [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:pixel_format
                                                         width:width
                                                        height:height
                                                     mipmapped:NO];
  td.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
  // td.storageMode = data_info->storage_mode;

  id<MTLTexture> texture = [device newTextureWithDescriptor:td];
  void *texture_id = (__bridge_retained void *)texture;
  NSLog(@"malloc texture %@", texture);
  video_buffer = new MetalTextureVideoBuffer(texture_id, width, height,
                                             *data_info, device_context);
  return BMF_LITE_StsOk;
}

int MetalTextureVideoBufferAllocator::releaseVideoBuffer(
    VideoBuffer *video_buffer) {
  return BMF_LITE_StsOk;
}

}

#endif