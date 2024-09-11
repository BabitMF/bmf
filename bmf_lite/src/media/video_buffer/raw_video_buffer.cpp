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

#include "raw_video_buffer.h"

namespace bmf_lite {

class RawVideoBufferImpl {
  public:
    void *data;
};

RawVideoBuffer::RawVideoBuffer(
    void *data, int width, int height, HardwareDataInfo hardware_data_info,
    std::shared_ptr<HWDeviceContext> device_context) {
    impl_ = std::make_shared<RawVideoBufferImpl>();

    impl_->data = data;

    width_ = width;
    height_ = height;
    device_context_ = device_context;
    hardware_data_info_ = hardware_data_info;
}

RawVideoBuffer::~RawVideoBuffer() {
    if (deleter_) {
        deleter_(this);
    }
}

int RawVideoBuffer::width() { return width_; }

int RawVideoBuffer::height() { return height_; }

void *RawVideoBuffer::data() { return impl_->data; }

std::shared_ptr<HWDeviceContext> RawVideoBuffer::getHWDeviceContext() {
    return device_context_;
}

HardwareDataInfo RawVideoBuffer::hardwareDataInfo() {
    return hardware_data_info_;
}

} // namespace bmf_lite
