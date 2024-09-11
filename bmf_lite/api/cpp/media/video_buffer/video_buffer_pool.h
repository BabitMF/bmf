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

#ifndef _BMFLITE_MEDIA_VIDEO_BUFFER_POOL_H_
#define _BMFLITE_MEDIA_VIDEO_BUFFER_POOL_H_

#include "media/video_buffer/hardware_device_context.h"
#include "media/video_buffer/pool.h"
#include "media/video_buffer/video_buffer.h"
#include "media/video_buffer/video_buffer_allocator.h"
#include <memory>
#include <vector>

namespace bmf_lite {

class BMF_LITE_EXPORT VideoBufferPool : public Pool {
  public:
    VideoBufferPool(int width, int height,
                    std::shared_ptr<VideoBufferAllocator> allocator,
                    HardwareDataInfo hardware_data_info,
                    std::shared_ptr<HWDeviceContext> device_context, int size);

    ~VideoBufferPool();

    MemoryType memoryType();

    inline int width() const { return width_; }

    inline int height() const { return height_; }

    std::shared_ptr<HWDeviceContext> hardwareDeviceContext() {
        return device_context_;
    }

    std::shared_ptr<VideoBuffer> acquireObject();

    bool reuseMemory(void *buffer);

  private:
    int width_ = 0;
    int height_ = 0;
    HardwareDataInfo hardware_data_info_;
    int size_ = 0;
    // std::mutex mutex_;
    std::vector<VideoBuffer *> allocated_buffer_;
    std::vector<std::shared_ptr<VideoBuffer>> valid_buffer_;
    std::shared_ptr<HWDeviceContext> device_context_;
    std::shared_ptr<VideoBufferAllocator> allocator_;
};

class BMF_LITE_EXPORT VideoBufferMultiPool : public Pool {
  public:
    VideoBufferMultiPool(std::shared_ptr<VideoBufferAllocator> allocator,
                         std::shared_ptr<HWDeviceContext> device_context,
                         int size);
    ~VideoBufferMultiPool();
    std::shared_ptr<VideoBuffer>
    acquireObject(int width, int height, HardwareDataInfo hardware_data_info);
    bool checkSame(std::shared_ptr<VideoBuffer> video_buffer, int width,
                   int height, HardwareDataInfo hardware_data_info);
    bool reuseMemory(void *buffer);

  private:
    int size_ = 0;
    // std::mutex mutex_;
    std::vector<VideoBuffer *> allocated_buffer_;
    std::vector<std::shared_ptr<VideoBuffer>> valid_buffer_;
    std::shared_ptr<HWDeviceContext> device_context_;
    std::shared_ptr<VideoBufferAllocator> allocator_;
};

} // namespace bmf_lite

#endif // _BMFLITE_MEDIA_VIDEO_BUFFER_POOL_H_