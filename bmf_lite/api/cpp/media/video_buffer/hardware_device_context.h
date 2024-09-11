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

#ifndef _BMFLITE_MEDIA_HARDWARE_DEVICE_CONTEXT_H_
#define _BMFLITE_MEDIA_HARDWARE_DEVICE_CONTEXT_H_

#include "common/bmf_common.h"
#include <memory>

namespace bmf_lite {

enum HWDeviceType {
    // unknown device type
    kHWDeviceTypeNone = 0,

    // egl context
    kHWDeviceTypeEGLCtx,

    // MTL device for iOS and macOS
    kHWDeviceTypeMTL,

};

class HWDeviceContext {
  public:
    // detroy the managed context.
    virtual ~HWDeviceContext() {}

    // return the device type
    virtual HWDeviceType deviceType() = 0;

    virtual int setCurrent() = 0;

    virtual int getContextInfo(void *&info) = 0;

    virtual std::shared_ptr<HWDeviceContext> storeCurrent() = 0;
};

typedef void (*FreeFunc)(void *);
struct EGLContextInfo {
    void *egl_display = NULL;
    void *egl_context = NULL;
    void *egl_read_surface = NULL;
    void *egl_draw_surface = NULL;
};

struct HardwareDeviceCreateInfo {
    HWDeviceType device_type;
    void *context_info = NULL;
};

struct HardwareDeviceSetInfo {
    HWDeviceType device_type;
    void *context_info = NULL;
    int owned = 0;
    FreeFunc free_func = NULL;
    void *user_data = NULL;
};

class BMF_LITE_EXPORT HWDeviceContextManager {
  public:
    static int
    getCurrentHwDeviceContext(const HWDeviceType device_type,
                              std::shared_ptr<HWDeviceContext> &context);

    static int createHwDeviceContext(HardwareDeviceCreateInfo *create_info,
                                     std::shared_ptr<HWDeviceContext> &context);

    static int setHwDeviceContext(HardwareDeviceSetInfo *set_info,
                                  std::shared_ptr<HWDeviceContext> &context);
};

class BMF_LITE_EXPORT HWDeviceContextGuard {
  public:
    HWDeviceContextGuard(std::shared_ptr<HWDeviceContext> hw_device_context) {
        if (hw_device_context != NULL) {
            old_context_ = hw_device_context->storeCurrent();
            hw_device_context->setCurrent();
        }
    }

    ~HWDeviceContextGuard() {
        if (old_context_ != NULL) {
            old_context_->setCurrent();
        }
    }

  private:
    std::shared_ptr<HWDeviceContext> old_context_ = NULL;
};

} // namespace bmf_lite

#endif // _BMFLITE_MEDIA_HARDWARE_DEVICE_CONTEXT_H_