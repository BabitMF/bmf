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

#define GL_GLEXT_PROTOTYPES
#define EGL_EGLEXT_PROTOTYPES

#include "egl_hardware_device_context.h"
#include "common/error_code.h"
#include <EGL/egl.h>
#include <iostream>

namespace bmf_lite {
EglHWDeviceContext::~EglHWDeviceContext() {
    if (owned_) {
        if (egl_display_ != NULL) {
            if (egl_read_surface_ != egl_draw_surface_ &&
                egl_draw_surface_ != NULL) {
                if (!eglDestroySurface(egl_display_, egl_draw_surface_)) {
                    // BMF_LITE_ERR("eglDestroySurface write surface failed");
                }
            }
            if (egl_read_surface_ != NULL) {
                if (!eglDestroySurface(egl_display_, egl_read_surface_)) {
                    // BMF_LITE_ERR("eglDestroySurface read surface failed");
                }
            }
            if (egl_context_ != NULL) {
                if (!eglDestroyContext(egl_display_, egl_context_)) {
                    // BMF_LITE_ERR("eglDestroyContext failed");
                }
            }

            if (!eglTerminate(egl_display_)) {
                // BMF_LITE_ERR("eglTerminate failed");
            }
        }
    } else {
        if (free_func_) {
            free_func_(user_data_);
        }
    }
}

EglHWDeviceContext::EglHWDeviceContext(void *egl_display, void *egl_context,
                                       void *egl_read_surface,
                                       void *egl_draw_surface, int owned) {
    egl_display_ = egl_display;
    egl_context_ = egl_context;
    egl_read_surface_ = egl_read_surface;
    egl_draw_surface_ = egl_draw_surface;
    owned_ = owned;
}

int EglHWDeviceContext::setFreeFunc(FreeFunc free_func, void *user_data) {
    free_func_ = free_func;
    user_data_ = user_data;
    return 0;
}

EglHWDeviceContext::EglHWDeviceContext() {}

int EglHWDeviceContext::create_context(void *shared_context) {
    egl_display_ = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (egl_display_ == EGL_NO_DISPLAY) {
        // BMF_LITE_ERR("egl get display failed");
        return BMF_LITE_EGLError;
    }
    int majorVersion;
    int minorVersion;
    if (!eglInitialize(egl_display_, &majorVersion, &minorVersion)) {
        // BMF_LITE_ERR("egl eglInitialize failed");
        return BMF_LITE_EGLError;
    }
    EGLint numConfigs;
    const EGLint configAttribs[] = {EGL_SURFACE_TYPE,
                                    EGL_PBUFFER_BIT,
                                    EGL_RENDERABLE_TYPE,
                                    EGL_OPENGL_ES2_BIT,
                                    EGL_RED_SIZE,
                                    8,
                                    EGL_GREEN_SIZE,
                                    8,
                                    EGL_BLUE_SIZE,
                                    8,
                                    EGL_ALPHA_SIZE,
                                    8,
                                    EGL_NONE};

    EGLConfig surfaceConfig;
    if (!eglChooseConfig(egl_display_, configAttribs, &surfaceConfig, 1,
                         &numConfigs)) {
        // BMF_LITE_ERR("egl eglChooseConfig failed");
        return BMF_LITE_EGLError;
    }

    static const EGLint contextAttribs[] = {EGL_CONTEXT_CLIENT_VERSION, 3,
                                            EGL_NONE};
    egl_context_ =
        eglCreateContext(egl_display_, surfaceConfig, NULL, contextAttribs);
    if (egl_context_ == NULL) {
        // BMF_LITE_ERR("egl eglCreateContext failed");
        if (!eglTerminate(egl_display_)) {
            // BMF_LITE_ERR("eglTerminate failed");
        }
        egl_display_ = NULL;
        return BMF_LITE_EGLError;
    }
    static const EGLint surfaceAttribs[] = {EGL_WIDTH, 1, EGL_HEIGHT, 1,
                                            EGL_NONE};
    egl_read_surface_ =
        eglCreatePbufferSurface(egl_display_, surfaceConfig, surfaceAttribs);
    if (egl_read_surface_ == NULL) {
        // BMF_LITE_ERR("egl eglCreatePbufferSurface failed");
        if (!eglDestroyContext(egl_display_, egl_context_)) {
            // BMF_LITE_ERR("eglDestroyContext failed");
        }
        egl_context_ = NULL;
        if (!eglTerminate(egl_display_)) {
            // BMF_LITE_ERR("eglTerminate failed");
        }
        egl_display_ = NULL;
        return BMF_LITE_EGLError;
    }
    egl_draw_surface_ = egl_read_surface_;
    owned_ = 1;
    return 0;
}

int EglHWDeviceContext::setCurrent() {
    if (egl_context_ != NULL && egl_read_surface_ != NULL &&
        egl_draw_surface_ != NULL && egl_display_ != NULL) {
        if (!eglBindAPI(EGL_OPENGL_ES_API)) {
            // BMF_LITE_ERR("eglBindAPI failed");
            return BMF_LITE_EGLError;
        }
        void *current_context = eglGetCurrentContext();
        if (current_context == egl_context_) {
            return 0;
        }
        if (!eglMakeCurrent(egl_display_, egl_draw_surface_, egl_read_surface_,
                            egl_context_)) {
            // BMF_LITE_ERR("eglMakeCurrent failed");
            return BMF_LITE_EGLError;
        }
    }
    return BMF_LITE_StsBadArg;
}

std::shared_ptr<HWDeviceContext> EglHWDeviceContext::storeCurrent() {
    std::shared_ptr<EglHWDeviceContext> context =
        std::make_shared<EglHWDeviceContext>();
    context->egl_display_ = eglGetCurrentDisplay();
    context->egl_draw_surface_ = eglGetCurrentSurface(EGL_DRAW);
    context->egl_read_surface_ = eglGetCurrentSurface(EGL_READ);
    context->egl_context_ = eglGetCurrentContext();
    return context;
}

int EglHWDeviceContext::getContextInfo(void *&context_info) {
    info.egl_context = egl_context_;
    info.egl_display = egl_display_;
    info.egl_draw_surface = egl_draw_surface_;
    info.egl_read_surface = egl_read_surface_;
    context_info = &info;
    return 0;
}
} // namespace bmf_lite

#endif