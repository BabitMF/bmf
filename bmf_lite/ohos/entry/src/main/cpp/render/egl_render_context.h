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
#ifndef NATIVEIMAGEDEMO_EGL_RENDER_CONTEXT_H
#define NATIVEIMAGEDEMO_EGL_RENDER_CONTEXT_H

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#include <GLES2/gl2ext.h>

namespace bmf_lite_demo {
class EglRenderContext {
  public:
    EglRenderContext();
    ~EglRenderContext() noexcept;

    // disallow copy and move
    EglRenderContext(const EglRenderContext &) = delete;
    void operator==(const EglRenderContext &) = delete;
    EglRenderContext(const EglRenderContext &&) = delete;
    void operator==(const EglRenderContext &&) = delete;

    bool Init();
    bool IsEglContextReady() const { return eglContext_ != EGL_NO_CONTEXT; }

    EGLDisplay GetEGLDisplay() const { return eglDisplay_; }
    EGLContext GetEGLContext() const { return eglContext_; }

    EGLSurface CreateEglSurface(EGLNativeWindowType surface,
                                const EGLint *attribList = nullptr);
    void DestroyEglSurface(EGLSurface surface);

    void MakeCurrent(EGLSurface surface) const;
    void SwapBuffers(EGLSurface surface) const;

    // 局部刷新可选接口
    EGLint QueryBufferAge(EGLSurface surface) const;
    void SetDamageRegion(EGLSurface surface, EGLint *damages, EGLint size);
    void SwapBuffersWithDamage(EGLSurface surface, const EGLint *damages,
                               EGLint size);

  private:
    void SetupEglExtensions();

  protected:
    EGLDisplay eglDisplay_ = EGL_NO_DISPLAY;
    EGLContext eglContext_ = EGL_NO_CONTEXT;
    EGLConfig config_ = nullptr;

    bool hasEglSurfacelessContext_ = false;
    bool hasEglBufferAge_ = false;
    bool hasEglPartialUpdate_ = false;
    PFNEGLSETDAMAGEREGIONKHRPROC eglSetDamageRegionFunc_ = nullptr;
    PFNEGLSWAPBUFFERSWITHDAMAGEKHRPROC eglSwapBuffersWithDamageFunc_ = nullptr;
    PFNEGLCREATEIMAGEKHRPROC eglCreateImageFunc_ = nullptr;
    PFNGLEGLIMAGETARGETTEXTURE2DOESPROC eglImageTargetTexture2DOESFunc_ =
        nullptr;
};
} // namespace bmf_lite_demo
#endif // NATIVEIMAGEDEMO_EGL_RENDER_CONTEXT_H
