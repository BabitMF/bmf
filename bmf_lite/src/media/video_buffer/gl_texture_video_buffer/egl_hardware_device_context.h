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

#ifndef _BMF_EGL_HARDWARE_DEVICE_CONTEXT_H_
#define _BMF_EGL_HARDWARE_DEVICE_CONTEXT_H_

#include "media/video_buffer/hardware_device_context.h"
#include <memory>
#include <stdio.h>

namespace bmf_lite {

class EglHWDeviceContext : public HWDeviceContext {
  public:
    std::shared_ptr<HWDeviceContext> storeCurrent();
    HWDeviceType deviceType() { return kHWDeviceTypeEGLCtx; }
    EglHWDeviceContext(void *egl_display, void *egl_context,
                       void *egl_read_surface, void *egl_draw_surface,
                       int owned);
    EglHWDeviceContext();
    ~EglHWDeviceContext();
    int setFreeFunc(FreeFunc free_func, void *user_data);
    int create_context(void *egl_context = NULL);
    int setCurrent();
    int getContextInfo(void *&info);
    EGLContextInfo info;
    void *egl_display_ = NULL;
    void *egl_context_ = NULL;
    void *egl_read_surface_ = NULL;
    void *egl_draw_surface_ = NULL;
    int owned_ = 0;

    FreeFunc free_func_ = NULL;
    void *user_data_ = NULL;
};

} // namespace bmf_lite

#endif