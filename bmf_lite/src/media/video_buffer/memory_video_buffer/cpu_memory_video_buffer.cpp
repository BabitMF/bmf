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

#ifdef BMF_LITE_ENABLE_CPUMEMORYBUFFER
#include "cpu_memory_video_buffer.h"

namespace bmf_lite {
CpuMemoryVideoBuffer::CpuMemoryVideoBuffer(
    void *memory_data, int width, int height,
    HardwareDataInfo hardware_data_info,
    std::shared_ptr<HWDeviceContext> device_context) {
    memory_data_ = memory_data;
    width_ = width;
    height_ = height;
    device_context_ = device_context;
    hardware_data_info_ = hardware_data_info;
}

CpuMemoryVideoBuffer::~CpuMemoryVideoBuffer() {
    if (deleter_) {
        deleter_(this);
    }
}

int CpuMemoryVideoBuffer::width() { return width_; }

int CpuMemoryVideoBuffer::height() { return height_; }

std::shared_ptr<HWDeviceContext> CpuMemoryVideoBuffer::getHWDeviceContext() {
    return device_context_;
}

MemoryType CpuMemoryVideoBuffer::memoryType() {
    return hardware_data_info_.mem_type;
}

HardwareDataInfo CpuMemoryVideoBuffer::hardwareDataInfo() {
    return hardware_data_info_;
}

} // namespace bmf_lite

#endif