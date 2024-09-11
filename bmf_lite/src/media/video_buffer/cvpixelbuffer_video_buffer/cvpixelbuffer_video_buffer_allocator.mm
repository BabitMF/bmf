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

#ifdef BMF_LITE_ENABLE_CVPIXELBUFFER

#include "cvpixelbuffer_video_buffer_allocator.h"
#include "common/error_code.h"
#include "cvpixelbuffer_video_buffer.h"
#import <AVFoundation/AVFoundation.h>
#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>

namespace bmf_lite {

CVPixelBufferVideoBufferAllocator::CVPixelBufferVideoBufferAllocator() {}
CVPixelBufferVideoBufferAllocator::~CVPixelBufferVideoBufferAllocator() {}

int CVPixelBufferVideoBufferAllocator::allocVideoBuffer(
    int width, int height, HardwareDataInfo *data_info,
    std::shared_ptr<HWDeviceContext> device_context,
    VideoBuffer *&video_buffer) {
  // HWDeviceContextGuard device_guard(device_context);
  CVPixelBufferRef cv_pixel_buffer;
  if (data_info->internal_format == BMF_LITE_CV_NV12) {
    NSDictionary *options = @{
      (__bridge NSString *)kCVPixelBufferIOSurfacePropertiesKey : @{},
    };
    int format = kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange;
    CVReturn result = CVPixelBufferCreate(
        kCFAllocatorDefault, width, height, format,
        (__bridge CFDictionaryRef)options, &cv_pixel_buffer);
    if (result != 0) {
      return BMF_LITE_COREVIDEO_ERROR;
    }
    video_buffer = new CVPixelBufferVideoBuffer(cv_pixel_buffer, width, height,
                                                *data_info, device_context);
    video_buffer->setDeleter([](VideoBuffer *video_buffer) {
      CVPixelBufferVideoBufferAllocator::releaseVideoBuffer(video_buffer);
    });
  }
  return 0;
}

int CVPixelBufferVideoBufferAllocator::releaseVideoBuffer(
    VideoBuffer *video_buffer) {
  std::shared_ptr<HWDeviceContext> device_context =
      video_buffer->getHWDeviceContext();
  CVPixelBufferVideoBuffer *cv_pixel_buffer =
      (CVPixelBufferVideoBuffer *)video_buffer;
  if (cv_pixel_buffer != NULL) {
    CVPixelBufferRelease((CVPixelBufferRef)(cv_pixel_buffer->cv_pixel_buffer_));
  }
  return 0;
} // namespace bmf_lite

}

#endif