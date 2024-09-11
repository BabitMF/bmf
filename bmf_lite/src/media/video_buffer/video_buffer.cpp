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

#include "common/error_code.h"
#include "media/video_buffer/video_buffer.h"
#include "media/video_buffer/raw_video_buffer.h"
#include "media/video_buffer/cvpixelbuffer_video_buffer/cvpixelbuffer_video_buffer.h"
#include "media/video_buffer/gl_texture_video_buffer/gl_texture_video_buffer.h"
#include "media/video_buffer/gl_texture_video_buffer/gl_texture_video_buffer_allocator.h"
#include "media/video_buffer/metal_texture_video_buffer/metal_texture_video_buffer.h"
#include "media/video_buffer/metal_texture_video_buffer/multi_metal_texture_video_buffer.h"
#include <media/video_buffer/memory_video_buffer/cpu_memory_video_buffer.h>

namespace bmf_lite {

int VideoBufferManager::createVideoBuffer(
    int width, int height, HardwareDataInfo *hardware_data_info,
    std::shared_ptr<HWDeviceContext> context,
    std::shared_ptr<VideoBuffer> &video_buffer) {

    std::shared_ptr<VideoBufferAllocator> allocator =
        AllocatorManager::getAllocator(hardware_data_info->mem_type, context);
    VideoBuffer *video_buffer_p;
    allocator->allocVideoBuffer(width, height, hardware_data_info, context,
                                video_buffer_p);
    std::shared_ptr<VideoBuffer> shared_buffer(video_buffer_p);
    video_buffer = shared_buffer;

    return BMF_LITE_StsOk;
}

int VideoBufferManager::createTextureVideoBufferFromExistingData(
    void *data, int width, int height, HardwareDataInfo *hardware_data_info,
    std::shared_ptr<HWDeviceContext> context, VideoBuffer::UserDeleter deleter,
    std::shared_ptr<VideoBuffer> &video_buffer) {

    if (hardware_data_info->mem_type == MemoryType::kOpenGLTexture2d ||
        hardware_data_info->mem_type == MemoryType::kOpenGLTextureExternalOes) {
#ifdef BMF_LITE_ENABLE_OPENGLTEXTUREBUFFER
        video_buffer = std::make_shared<GlTextureVideoBuffer>(
            (long)(data), width, height, *hardware_data_info, context);
        video_buffer->setDeleter(deleter);
#endif
    } else if (hardware_data_info->mem_type == MemoryType::kMetalTexture) {
#ifdef BMF_LITE_ENABLE_METALBUFFER
        video_buffer = std::make_shared<MetalTextureVideoBuffer>(
            data, width, height, *hardware_data_info, context);
        video_buffer->setDeleter(deleter);
#endif
    } else if (hardware_data_info->mem_type == MemoryType::kMultiMetalTexture) {
#ifdef BMF_LITE_ENABLE_METALBUFFER
        video_buffer = std::make_shared<MultiMetalTextureVideoBuffer>(
            data, width, height, *hardware_data_info, context);
        video_buffer->setDeleter(deleter);
#endif
    } else if (hardware_data_info->mem_type == MemoryType::kCVPixelBuffer) {
#ifdef BMF_LITE_ENABLE_METALBUFFER
        video_buffer = std::make_shared<CVPixelBufferVideoBuffer>(
            data, width, height, *hardware_data_info, context);
        video_buffer->setDeleter(deleter);
#endif
    } else if (hardware_data_info->mem_type == MemoryType::kRaw) {
        video_buffer = std::make_shared<RawVideoBuffer>(
            data, width, height, *hardware_data_info, context);
        video_buffer->setDeleter(deleter);
    } else if (hardware_data_info->mem_type == MemoryType::kByteMemory) {
#ifdef BMF_LITE_ENABLE_CPUMEMORYBUFFER
        video_buffer = std::make_shared<CpuMemoryVideoBuffer>(
            data, width, height, *hardware_data_info, context);
        video_buffer->setDeleter(deleter);
#endif
    }
    return BMF_LITE_StsOk;
}

} // namespace bmf_lite