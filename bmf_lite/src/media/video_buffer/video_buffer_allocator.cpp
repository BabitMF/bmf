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

#include <media/video_buffer/cvpixelbuffer_video_buffer/cvpixelbuffer_video_buffer_allocator.h>
#include <media/video_buffer/gl_texture_video_buffer/gl_texture_video_buffer_allocator.h>
#include <media/video_buffer/metal_texture_video_buffer/metal_texture_video_buffer_allocator.h>
#include <media/video_buffer/metal_texture_video_buffer/multi_metal_texture_video_buffer_allocator.h>
#include <media/video_buffer/memory_video_buffer/cpu_memory_video_buffer_allocator.h>
#include <media/video_buffer/video_buffer_allocator.h>

namespace bmf_lite {

std::shared_ptr<VideoBufferAllocator> AllocatorManager::getAllocator(
    const MemoryType memory_type,
    std::shared_ptr<HWDeviceContext> device_context) {

    if (memory_type == MemoryType::kOpenGLTexture2d) {
#ifdef BMF_LITE_ENABLE_OPENGLTEXTUREBUFFER
        return std::make_shared<GlTextureVideoBufferAllocator>();
#endif
    } else if (memory_type == MemoryType::kMetalTexture) {
#ifdef BMF_LITE_ENABLE_METALBUFFER
        return std::make_shared<MetalTextureVideoBufferAllocator>();
#endif
    } else if (memory_type == MemoryType::kMultiMetalTexture) {
#ifdef BMF_LITE_ENABLE_METALBUFFER
        return std::make_shared<MultiMetalTextureVideoBufferAllocator>();
#endif
    } else if (memory_type == MemoryType::kCVPixelBuffer) {
#ifdef BMF_LITE_ENABLE_CVPIXELBUFFER
        return std::make_shared<CVPixelBufferVideoBufferAllocator>();
#endif
    } else if (memory_type == MemoryType::kByteMemory) {
#ifdef BMF_LITE_ENABLE_CPUMEMORYBUFFER
        return std::make_shared<CpuMemoryVideoBufferAllocator>();
#endif
    }
    return NULL;
}

} // namespace bmf_lite