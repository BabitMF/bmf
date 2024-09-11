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
#include "render_thread.h"

#include "common/common.h"
#include <hilog/log.h>

namespace bmf_lite_demo {

RenderThread::RenderThread() { Start(); }

RenderThread::~RenderThread() noexcept {
    if (algorithm_running_) {
        if (algorithm_) {
            StopAlgorithm();
        }
    }
    PostTask([this](EglRenderContext &renderContext) {
        CleanGLResources();
        DestroyNativeImage();
        UpdateNativeWindow(nullptr, 0, 0);
        DestroyNativeVsync();
        DestroyRenderContext();
    });

    running_ = false;
    if (thread_.joinable()) {
        thread_.join();
    }
}

bool RenderThread::InitRenderContext() {
    renderContext_ = std::make_unique<EglRenderContext>();
    return renderContext_->Init();
}

void RenderThread::DestroyRenderContext() { renderContext_.reset(); }

void RenderThread::CleanGLResources() {
    if (out_texture_2d_ != GL_NONE) {
        glDeleteTextures(1, &out_texture_2d_);
        out_texture_2d_ = GL_NONE;
    }
}

bool RenderThread::CreateGLResources() {
    oesTo2dRenderer_ = std::make_unique<OesTo2dRenderer>();
    int ret = oesTo2dRenderer_->init();
    if (ret != 0) {
        return false;
    }
    splitScreenRenderer_ = std::make_unique<SplitScreenRenderer>();
    ret = splitScreenRenderer_->init();
    return ret == 0;
}

void RenderThread::UpdateNativeWindow(void *window, uint64_t width,
                                      uint64_t height) {
    OH_LOG_Print(LOG_APP, LOG_DEBUG, LOG_PRINT_DOMAIN, "RenderThread",
                 "UpdateNativeWindow.");
    auto nativeWindow = reinterpret_cast<OHNativeWindow *>(window);
    PostTask([this, nativeWindow, width,
              height](EglRenderContext &renderContext) {
        if (nativeWindow_ != nativeWindow) {
            if (nativeWindow_ != nullptr) {
                (void)OH_NativeWindow_NativeObjectUnreference(nativeWindow_);
            }
            nativeWindow_ = nativeWindow;
            if (nativeWindow_ != nullptr) {
                (void)OH_NativeWindow_NativeObjectReference(nativeWindow_);
                nativeWindowWidth_ = width;
                nativeWindowHeight_ = height;
            }

            if (eglSurface_ != EGL_NO_SURFACE) {
                renderContext_->DestroyEglSurface(eglSurface_);
                eglSurface_ = EGL_NO_SURFACE;
                CleanGLResources();
            }
        }

        if (nativeWindow_ != nullptr) {
            (void)OH_NativeWindow_NativeWindowHandleOpt(
                nativeWindow_, SET_BUFFER_GEOMETRY, static_cast<int>(width),
                static_cast<int>(height));
            if (eglSurface_ == EGL_NO_SURFACE) {
                eglSurface_ = renderContext_->CreateEglSurface(
                    static_cast<EGLNativeWindowType>(nativeWindow_));
            }
            if (eglSurface_ == EGL_NO_SURFACE) {
                OH_LOG_Print(LOG_APP, LOG_ERROR, LOG_PRINT_DOMAIN,
                             "RenderThread", "CreateEglSurface failed.");
                return;
            }
            renderContext_->MakeCurrent(eglSurface_);
            CreateGLResources();
        }
    });
}

void RenderThread::StartAlgorithm(AlgorithmEnum algorithm) {
    OH_LOG_Print(LOG_APP, LOG_DEBUG, LOG_PRINT_DOMAIN, "RenderThread",
                 "StartAlgorithm.");
    AlgorithmContext algorithmContext{renderContext_->GetEGLDisplay(),
                                      renderContext_->GetEGLContext(),
                                      eglSurface_, algorithm};
    PostTask([this, algorithmContext](EglRenderContext &renderContext) {
        if (!algorithm_running_) {
            algorithm_ = std::make_shared<Algorithm>(
                algorithmContext.egl_display, algorithmContext.egl_surface,
                algorithmContext.egl_context, algorithmContext.algorithm);
            algorithm_running_ = true;
        }
    });
}

void RenderThread::StopAlgorithm() {
    OH_LOG_Print(LOG_APP, LOG_DEBUG, LOG_PRINT_DOMAIN, "RenderThread",
                 "StopAlgorithm.");
    PostTask([this](EglRenderContext &renderContext) {
        algorithm_running_ = false;
        algorithm_.reset();
    });
}

void RenderThread::UpdateSplitRatio(float ratio) {
    OH_LOG_Print(LOG_APP, LOG_DEBUG, LOG_PRINT_DOMAIN, "RenderThread",
                 "UpdateProportion %{public}f", ratio);
    splitRatio_ = ratio;
}

void RenderThread::SetScene(int scene) { scene_ = scene; }

void RenderThread::Start() {
    if (running_) {
        return;
    }

    running_ = true;
    thread_ = std::thread([this]() {
        ThreadMainLoop();
        // 确保renderContext的创建和销毁都在渲染线程中执行
        CleanGLResources();
        DestroyNativeImage();
        UpdateNativeWindow(nullptr, 0, 0);
        DestroyNativeVsync();
        DestroyRenderContext();
        running_ = false;
    });
}

void RenderThread::OnVsync(long long timestamp, void *data) {
    //     OH_LOG_Print(LOG_APP, LOG_DEBUG, LOG_PRINT_DOMAIN, "RenderThread",
    //     "OnVsync %{public}llu.", timestamp);
    auto renderThread = reinterpret_cast<RenderThread *>(data);
    if (renderThread == nullptr) {
        return;
    }

    renderThread->vSyncCnt_++;
    renderThread->wakeUpCond_.notify_one();
}

bool RenderThread::InitNativeVsync() {
    nativeVsync_ = OH_NativeVSync_Create(DEMO_NAME, strlen(DEMO_NAME));
    if (nativeVsync_ == nullptr) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, LOG_PRINT_DOMAIN, "RenderThread",
                     "Create NativeVSync failed.");
        return false;
    }
    (void)OH_NativeVSync_RequestFrame(nativeVsync_, &RenderThread::OnVsync,
                                      this);
    return true;
}

OHNativeWindow *RenderThread::GetNativeWindow() { return nativeWindow_; }

void RenderThread::DestroyNativeVsync() {
    if (nativeVsync_ != nullptr) {
        OH_NativeVSync_Destroy(nativeVsync_);
        nativeVsync_ = nullptr;
    }
}

void RenderThread::OnNativeImageFrameAvailable(void *data) {
    //     OH_LOG_Print(LOG_APP, LOG_DEBUG, LOG_PRINT_DOMAIN, "RenderThread",
    //     "OnNativeImageFrameAvailable.");
    auto renderThread = reinterpret_cast<RenderThread *>(data);
    if (renderThread == nullptr) {
        return;
    }
    renderThread->availableFrameCnt_++;
    renderThread->wakeUpCond_.notify_one();
}

bool RenderThread::CreateNativeImage() {
    glGenTextures(1, &nativeImageTexId_);
    glBindTexture(GL_TEXTURE_EXTERNAL_OES, nativeImageTexId_);
    // set the texture wrapping parameters
    glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_WRAP_T, GL_REPEAT);
    // set texture filtering parameters
    glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    nativeImage_ =
        OH_NativeImage_Create(nativeImageTexId_, GL_TEXTURE_EXTERNAL_OES);
    if (nativeImage_ == nullptr) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, LOG_PRINT_DOMAIN, "RenderThread",
                     "OH_NativeImage_Create failed.");
        return false;
    }
    int ret = 0;
    {
        std::lock_guard<std::mutex> lock(nativeImageSurfaceIdMutex_);
        ret = OH_NativeImage_GetSurfaceId(nativeImage_, &nativeImageSurfaceId_);
    }
    if (ret != 0) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, LOG_PRINT_DOMAIN, "RenderThread",
                     "OH_NativeImage_GetSurfaceId failed, ret is %{public}d.",
                     ret);
        return false;
    }
    {
        std::lock_guard<std::mutex> lock(nativeImageNativeWindowMutex_);
        nativeImageNativeWindow_ =
            OH_NativeImage_AcquireNativeWindow(nativeImage_);
    }
    if (nativeImageNativeWindow_ == nullptr) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, LOG_PRINT_DOMAIN, "RenderThread",
                     "OH_NativeImage_AcquireNativeWindow failed");
        return false;
    }

    nativeImageFrameAvailableListener_.context = this;
    nativeImageFrameAvailableListener_.onFrameAvailable =
        &RenderThread::OnNativeImageFrameAvailable;
    ret = OH_NativeImage_SetOnFrameAvailableListener(
        nativeImage_, nativeImageFrameAvailableListener_);
    if (ret != 0) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, LOG_PRINT_DOMAIN, "RenderThread",
                     "OH_NativeImage_SetOnFrameAvailableListener failed, ret "
                     "is %{public}d.",
                     ret);
        return false;
    }

    return true;
}

void RenderThread::DestroyNativeImage() {
    if (nativeImageTexId_ != 0U) {
        glDeleteTextures(1, &nativeImageTexId_);
        nativeImageTexId_ = 0U;
    }

    if (nativeImage_ != nullptr) {
        (void)OH_NativeImage_UnsetOnFrameAvailableListener(nativeImage_);
        OH_NativeImage_Destroy(&nativeImage_);
        nativeImage_ = nullptr;
    }
    nativeImageSurfaceId_ = 0;
    nativeImageNativeWindow_ = nullptr;
}

void RenderThread::DestroyAlgorithm() {
    if (algorithm_) {
        algorithm_ = nullptr;
    }
    algorithm_running_ = false;
}

void RenderThread::ThreadMainLoop() {
    threadId_ = std::this_thread::get_id();
    if (!InitRenderContext()) {
        return;
    }
    if (!InitNativeVsync()) {
        return;
    }
    if (!CreateNativeImage()) {
        return;
    }
    while (running_) {
        {
            //             OH_LOG_Print(LOG_APP, LOG_DEBUG, LOG_PRINT_DOMAIN,
            //             "RenderThread", "Waiting for vsync.");
            std::unique_lock<std::mutex> lock(wakeUpMutex_);
            wakeUpCond_.wait(lock, [this]() {
                return wakeUp_ || vSyncCnt_ > 0 || availableFrameCnt_ > 0;
            });
            wakeUp_ = false;
            vSyncCnt_--;
            (void)OH_NativeVSync_RequestFrame(nativeVsync_,
                                              &RenderThread::OnVsync, this);
        }

        //         OH_LOG_Print(LOG_APP, LOG_DEBUG, LOG_PRINT_DOMAIN,
        //         "RenderThread", "Executing tasks.");
        std::vector<RenderTask> tasks;
        {
            std::lock_guard<std::mutex> lock(taskMutex_);
            tasks.swap(tasks_);
        }
        for (const auto &task : tasks) {
            task(*renderContext_);
        }

        if (availableFrameCnt_ <= 0) {
            continue;
        }
        DrawImage();
        availableFrameCnt_--;
    }
}

void RenderThread::PostTask(const RenderTask &task) {
    if (!running_) {
        OH_LOG_Print(LOG_APP, LOG_WARN, LOG_PRINT_DOMAIN, "RenderThread",
                     "PostTask failed: RenderThread is not running");
        return;
    }

    {
        std::lock_guard<std::mutex> lock(taskMutex_);
        tasks_.push_back(task);
    }

    if (std::this_thread::get_id() != threadId_) {
        std::lock_guard<std::mutex> lock(wakeUpMutex_);
        wakeUp_ = true;
        wakeUpCond_.notify_one();
    }
}

void RenderThread::DrawImage() {
    //     OH_LOG_Print(LOG_APP, LOG_DEBUG, LOG_PRINT_DOMAIN, "RenderThread",
    //     "DrawImage.");
    if (eglSurface_ == EGL_NO_SURFACE) {
        OH_LOG_Print(LOG_APP, LOG_WARN, LOG_PRINT_DOMAIN, "RenderThread",
                     "eglSurface_ is EGL_NO_SURFACE");
        return;
    }

    renderContext_->MakeCurrent(eglSurface_);
    int32_t ret = OH_NativeImage_UpdateSurfaceImage(nativeImage_);
    if (ret != 0) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, LOG_PRINT_DOMAIN, "RenderThread",
                     "OH_NativeImage_UpdateSurfaceImage failed, ret: "
                     "%{public}d, texId: %{public}d",
                     ret, nativeImageTexId_);
        return;
    }

    //     float matrix[16];
    //     ret = OH_NativeImage_GetTransformMatrix(nativeImage_, matrix);
    float matrix[16] = {1, 0, 0, 0, 0, -1, 0, 0, 0, -1, 1, 0, 0, 0, 0, 1};
    if (scene_ == 1) {
        matrix[0] = 0;
        matrix[1] = -1;
        matrix[4] = 1;
        matrix[5] = 0;
        matrix[8] = -1;
        matrix[9] = 0;
        matrix[10] = 0;
    }

    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    if (out_texture_2d_ == GL_NONE) {
        glGenTextures(1, &out_texture_2d_);
        glBindTexture(GL_TEXTURE_2D, out_texture_2d_);
        // set the texture wrapping parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        // set the texture filtering parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, nativeWindowWidth_,
                     nativeWindowHeight_, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
        glBindTexture(GL_TEXTURE_2D, GL_NONE);
    }
    std::vector<float> matrix_vec(matrix, matrix + 16);
    oesTo2dRenderer_->setMatrix(matrix_vec);

    if (algorithm_running_) {
        oesTo2dRenderer_->process(nativeImageTexId_, out_texture_2d_,
                                  nativeWindowWidth_, nativeWindowHeight_);
        GLuint out_texture = 0;
        size_t out_width = 0;
        size_t out_height = 0;
        if (algorithm_) {
            int res = algorithm_->processVideoFrame(
                out_texture_2d_, nativeWindowWidth_, nativeWindowHeight_);
            if (res == 0) {
                algorithm_->getVideoFrameOutput(out_texture, out_width,
                                                out_height);
            }
        }
        if (out_texture != 0) {
            float defaultMatrix[16] = {1, 0, 0, 0, 0, 1, 0, 0,
                                       0, 0, 1, 0, 0, 0, 0, 1};
            std::vector<float> matrix_vec2(defaultMatrix, defaultMatrix + 16);
            splitScreenRenderer_->setMatrix(matrix_vec2);
            splitScreenRenderer_->setDisplayDivider(1);
            splitScreenRenderer_->setSplitRatio(splitRatio_);
            splitScreenRenderer_->drawToScreen(out_texture_2d_, out_texture,
                                               nativeWindowWidth_,
                                               nativeWindowHeight_);
        } else {
            oesTo2dRenderer_->drawToScreen(
                nativeImageTexId_, nativeWindowWidth_, nativeWindowHeight_);
        }
    } else {
        oesTo2dRenderer_->drawToScreen(nativeImageTexId_, nativeWindowWidth_,
                                       nativeWindowHeight_);
    }

    renderContext_->SwapBuffers(eglSurface_);
}
} // namespace bmf_lite_demo
