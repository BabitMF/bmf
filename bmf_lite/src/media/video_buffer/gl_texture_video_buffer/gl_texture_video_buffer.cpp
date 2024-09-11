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

#ifdef BMF_LITE_ENABLE_OPENGLTEXTUREBUFFER
#include "gl_texture_video_buffer.h"

namespace bmf_lite {
GlTextureVideoBuffer::GlTextureVideoBuffer(
    int texture_id, int width, int height, HardwareDataInfo hardware_data_info,
    std::shared_ptr<HWDeviceContext> device_context) {
    texture_id_ = texture_id;
    width_ = width;
    height_ = height;
    device_context_ = device_context;
    hardware_data_info_ = hardware_data_info;
}

GlTextureVideoBuffer::~GlTextureVideoBuffer() {
    if (deleter_) {
        deleter_(this);
    }
}

int GlTextureVideoBuffer::width() { return width_; }

int GlTextureVideoBuffer::height() { return height_; }

int GlTextureVideoBuffer::getTextureId() { return texture_id_; }

std::shared_ptr<HWDeviceContext> GlTextureVideoBuffer::getHWDeviceContext() {
    return device_context_;
}

MemoryType GlTextureVideoBuffer::memoryType() {
    return hardware_data_info_.mem_type;
}

HardwareDataInfo GlTextureVideoBuffer::hardwareDataInfo() {
    return hardware_data_info_;
}

} // namespace bmf_lite

#endif