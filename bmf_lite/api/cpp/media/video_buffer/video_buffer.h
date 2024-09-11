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

#ifndef _BMFLITE_MEDIA_VIDEO_BUFFER_H_
#define _BMFLITE_MEDIA_VIDEO_BUFFER_H_

#include "format.h"
#include "hardware_device_context.h"
#include "pool.h"

namespace bmf_lite {

class VideoBuffer {
  public:
    // detroy the managed context.
    typedef void (*UserDeleter)(VideoBuffer *);
    VideoBuffer() {}

    VideoBuffer(void *data, int width, int height, MemoryType memory_type,
                std::shared_ptr<HWDeviceContext> context);

    virtual ~VideoBuffer() {};

    virtual std::shared_ptr<HWDeviceContext> getHWDeviceContext() = 0;

    virtual int width() = 0;

    virtual int height() = 0;

    virtual void *data() = 0;

    virtual MemoryType memoryType() = 0;

    virtual HardwareDataInfo hardwareDataInfo() = 0;

    virtual void setDeleter(UserDeleter deleter) { deleter_ = deleter; }

    void setPool(Pool *pool) { pool_ = pool; }

    bool poolRetain() {
        if (pool_) {
            pool_->reuseMemory((void *)this);
            return true;
        }
        return false;
    }

    std::shared_ptr<HWDeviceContext> device_context_ = NULL;
    int width_ = 0;
    int height_ = 0;
    MemoryType memory_type_;
    Pool *pool_ = NULL;
    UserDeleter deleter_ = NULL;
    HardwareDataInfo hardware_data_info_;
};

struct VideoTextureList {
    VideoBuffer **texture_list;
    int num;
};

struct VideoTextures {
    void **texture_list;
    int num;
};

class BMF_LITE_EXPORT VideoBufferManager {
  public:
    static int createVideoBuffer(int width, int height,
                                 HardwareDataInfo *hardware_data_info,
                                 std::shared_ptr<HWDeviceContext> context,
                                 std::shared_ptr<VideoBuffer> &video_buffer);

    static int createTextureVideoBufferFromExistingData(
        void *data, int width, int height, HardwareDataInfo *hardware_data_info,
        std::shared_ptr<HWDeviceContext> context,
        VideoBuffer::UserDeleter deleter,
        std::shared_ptr<VideoBuffer> &video_buffer);
};

}; // namespace bmf_lite

#endif // _BMFLITE_MEDIA_VIDEO_BUFFER_H_