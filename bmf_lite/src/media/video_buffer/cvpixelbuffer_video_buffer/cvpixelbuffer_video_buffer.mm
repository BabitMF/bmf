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

#include "cvpixelbuffer_video_buffer.h"

#ifdef BMF_LITE_ENABLE_CVPIXELBUFFER

namespace bmf_lite {
CVPixelBufferVideoBuffer::CVPixelBufferVideoBuffer(
    void *cv_pixel_buffer, int width, int height, HardwareDataInfo data_info,
    std::shared_ptr<HWDeviceContext> device_context) {
  cv_pixel_buffer_ = cv_pixel_buffer;
  width_ = width;
  height_ = height;
  hardware_data_info_ = data_info;
  device_context_ = device_context;
};

CVPixelBufferVideoBuffer::~CVPixelBufferVideoBuffer() {
  if (deleter_) {
    deleter_(this);
  }
}

int CVPixelBufferVideoBuffer::width() { return width_; }

int CVPixelBufferVideoBuffer::height() { return height_; }

std::shared_ptr<HWDeviceContext>
CVPixelBufferVideoBuffer::getHWDeviceContext() {
  return device_context_;
}

HardwareDataInfo CVPixelBufferVideoBuffer::hardwareDataInfo() {
  return hardware_data_info_;
}

} // namespace bmf_lite

#endif