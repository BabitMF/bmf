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
#define GL_GLEXT_PROTOTYPES
#define EGL_EGLEXT_PROTOTYPES

#include "cpu_memory_video_buffer_allocator.h"
#include "cpu_memory_video_buffer.h"

#include <EGL/egl.h>
#include <GLES3/gl3.h>
#include <iostream>

namespace bmf_lite {
#define GL_CHECK_RETURN(FUNC)                                                  \
    FUNC;                                                                      \
    {                                                                          \
        GLenum glError = glGetError();                                         \
        if (glError != GL_NO_ERROR) {                                          \
            std::cout << "Call " << #FUNC << "failed error code:" << glError   \
                      << std::endl;                                            \
            return -1;                                                         \
        }                                                                      \
    }

CpuMemoryVideoBufferAllocator::CpuMemoryVideoBufferAllocator() {}

CpuMemoryVideoBufferAllocator::~CpuMemoryVideoBufferAllocator() {}

int CpuMemoryVideoBufferAllocator::allocVideoBuffer(
    int width, int height, HardwareDataInfo *data_info,
    std::shared_ptr<HWDeviceContext> device_context,
    VideoBuffer *&video_buffer) {
    void *cpu_memory_ptr = nullptr;

    switch (data_info->internal_format) {
    case CPU_RGB:
        cpu_memory_ptr = (void *)(new unsigned char[width * height * 3]);
        break;
    case CPU_RGBA:
        cpu_memory_ptr = (void *)(new unsigned char[width * height * 4]);
        break;
    case CPU_RGBFLOAT:
        cpu_memory_ptr = (void *)(new float[width * height * 3]);
        break;
    case CPU_RGBAFLOAT:
        cpu_memory_ptr = (void *)(new float[width * height * 4]);
        break;
    default:
        cpu_memory_ptr = (void *)(new unsigned char[width * height * 3]);
        break;
    }

    video_buffer = new CpuMemoryVideoBuffer(cpu_memory_ptr, width, height,
                                            *data_info, device_context);
    if (video_buffer != NULL) {
        video_buffer->setDeleter([](VideoBuffer *video_buffer) {
            CpuMemoryVideoBufferAllocator::releaseVideoBuffer(video_buffer);
        });
        return 0;
    }
    return 0;
}

int CpuMemoryVideoBufferAllocator::releaseVideoBuffer(
    VideoBuffer *video_buffer) {
    std::shared_ptr<HWDeviceContext> device_context =
        video_buffer->getHWDeviceContext();
    if (video_buffer != NULL &&
        video_buffer->memoryType() == MemoryType::kByteMemory) {
        CpuMemoryVideoBuffer *cpu_video_buffer =
            (CpuMemoryVideoBuffer *)video_buffer;
        void *cpu_memory_ptr = cpu_video_buffer->data();
        if (cpu_memory_ptr != 0) {
            delete[] cpu_memory_ptr;
            cpu_memory_ptr = NULL;
        }
    }
    video_buffer = NULL;
    return 0;
}

} // namespace bmf_lite
#endif