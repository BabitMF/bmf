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
#ifndef BMFLITE_ALGORITHM_H
#define BMFLITE_ALGORITHM_H

#include <EGL/egl.h>
#include <GLES2/gl2.h>
#include <cstddef>

#include "algorithm/bmf_algorithm.h"

namespace bmf_lite_demo {

class Algorithm {
  public:
    Algorithm(EGLDisplay eglDisplay, EGLSurface readSurface,
              EGLContext eglContext, int algorithmType);
    Algorithm(Algorithm &) = delete;
    int processVideoFrame(GLuint textureId, size_t width, size_t height);
    int getVideoFrameOutput(GLuint &textureId, size_t &width, size_t &height);
    ~Algorithm();

  private:
    bmf_lite::IAlgorithm *algorithm_;

    int algorithmType_;

    EGLDisplay eglDisplay_ = EGL_NO_DISPLAY;
    EGLSurface eglSurface_ = EGL_NO_SURFACE;
    EGLContext eglContext_ = EGL_NO_CONTEXT;

    bool valid_ = false;
};

} // namespace bmf_lite_demo

#endif // BMFLITE_ALGORITHM_H
