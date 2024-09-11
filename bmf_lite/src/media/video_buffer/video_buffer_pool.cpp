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

#include "media/video_buffer/video_buffer_pool.h"

namespace bmf_lite {

VideoBufferPool::VideoBufferPool(
    int width, int height, std::shared_ptr<VideoBufferAllocator> allocator,
    HardwareDataInfo hardware_data_info,
    std::shared_ptr<HWDeviceContext> device_context, int size)
    : width_(width), height_(height), device_context_(device_context),
      size_(size), hardware_data_info_(hardware_data_info),
      allocator_(allocator) {}

VideoBufferPool::~VideoBufferPool() {
    for (int i = 0; i < allocated_buffer_.size(); i++) {
        allocated_buffer_[i]->setPool(NULL);
    }
}

MemoryType VideoBufferPool::memoryType() {
    if (device_context_ != NULL) {
        if (device_context_->deviceType() == kHWDeviceTypeEGLCtx) {
            return MemoryType::kOpenGLTexture2d;
        }
    }
    return MemoryType::kUnknown;
}

std::shared_ptr<VideoBuffer> VideoBufferPool::acquireObject() {
    if (valid_buffer_.size() > 0) {
        std::shared_ptr<VideoBuffer> buffer = valid_buffer_[0];
        valid_buffer_.erase(valid_buffer_.begin());
        return buffer;
    } else {
        if (allocated_buffer_.size() >= size_) {
            return NULL;
        } else {
            VideoBuffer *buffer = NULL;
            if (allocator_->allocVideoBuffer(width_, height_,
                                             &hardware_data_info_,
                                             device_context_, buffer) == 0) {
                std::shared_ptr<VideoBuffer> shared_buffer(
                    buffer, [](VideoBuffer *obj) {
                        if (obj && !obj->poolRetain()) {
                            delete obj;
                        }
                    });
                shared_buffer->setPool(this);
                allocated_buffer_.push_back(buffer);
                return shared_buffer;
            }
        }
    }
    return NULL;
}

bool VideoBufferPool::reuseMemory(void *buffer) {
    std::shared_ptr<VideoBuffer> share_buffer(
        (VideoBuffer *)buffer, [](VideoBuffer *obj) {
            if (obj && !obj->poolRetain()) {
                delete obj;
            }
        });
    share_buffer->setPool(this);
    valid_buffer_.push_back(share_buffer);
    return true;
}

VideoBufferMultiPool::VideoBufferMultiPool(
    std::shared_ptr<VideoBufferAllocator> allocator,
    std::shared_ptr<HWDeviceContext> device_context, int size) {
    device_context_ = device_context;
    size_ = size;
    allocator_ = allocator;
}

VideoBufferMultiPool::~VideoBufferMultiPool() {
    for (int i = 0; i < allocated_buffer_.size(); i++) {
        allocated_buffer_[i]->setPool(NULL);
    }
}

bool VideoBufferMultiPool::checkSame(std::shared_ptr<VideoBuffer> video_buffer,
                                     int width, int height,
                                     HardwareDataInfo hardware_data_info) {
    if (video_buffer->width() == width && video_buffer->height() == height &&
        video_buffer->hardwareDataInfo() == hardware_data_info) {
        return true;
    } else {
        return false;
    }
    return false;
}

std::shared_ptr<VideoBuffer>
VideoBufferMultiPool::acquireObject(int width, int height,
                                    HardwareDataInfo hardware_data_info) {
    for (int i = 0; i < valid_buffer_.size(); i++) {
        if (checkSame(valid_buffer_[i], width, height, hardware_data_info)) {
            std::shared_ptr<VideoBuffer> buffer = valid_buffer_[i];
            valid_buffer_.erase(valid_buffer_.begin() + i);
            return buffer;
        }
    }
    if (allocated_buffer_.size() >= size_ && valid_buffer_.size() > 0) {
        std::shared_ptr<VideoBuffer> buffer = valid_buffer_[0];
        VideoBuffer *ptr = buffer.get();
        for (int i = 0; i < allocated_buffer_.size(); i++) {
            if (allocated_buffer_[i] == ptr) {
                allocated_buffer_[i]->setPool(NULL);
                allocated_buffer_.erase(allocated_buffer_.begin() + i);
                break;
            }
        }
        valid_buffer_.erase(valid_buffer_.begin());
    }
    {
        if (allocated_buffer_.size() >= size_) {
            return NULL;
        } else {
            VideoBuffer *buffer = NULL;
            if (allocator_->allocVideoBuffer(width, height, &hardware_data_info,
                                             device_context_, buffer) == 0) {
                std::shared_ptr<VideoBuffer> shared_buffer(
                    buffer, [](VideoBuffer *obj) {
                        if (obj && !obj->poolRetain()) {
                            delete obj;
                        }
                    });
                shared_buffer->setPool(this);
                allocated_buffer_.push_back(buffer);
                return shared_buffer;
            };
        }
    }
    return NULL;
}
bool VideoBufferMultiPool::reuseMemory(void *buffer) {
    std::shared_ptr<VideoBuffer> share_buffer(
        (VideoBuffer *)buffer, [](VideoBuffer *obj) {
            if (obj && !obj->poolRetain()) {
                delete obj;
            }
        });
    share_buffer->setPool(this);
    valid_buffer_.push_back(share_buffer);
    return true;
}

} // namespace bmf_lite
