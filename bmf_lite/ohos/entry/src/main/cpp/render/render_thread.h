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
#ifndef NATIVEIMAGEDEMO_RENDER_THREAD_H
#define NATIVEIMAGEDEMO_RENDER_THREAD_H

#include <thread>
#include <vector>

#include <native_image/native_image.h>
#include <native_window/external_window.h>
#include <native_vsync/native_vsync.h>

#include "egl_render_context.h"
#include "render/oes_to_2d_renderer.h"
#include "render/split_screen_renderer.h"
#include "shader_program.h"

#include "algorithm.h"
#include "algorithm_context.h"

namespace bmf_lite_demo {
using RenderTask = std::function<void(EglRenderContext &renderContext)>;

class RenderThread {
  public:
    RenderThread();
    ~RenderThread() noexcept;

    // @param window - NativeWindow created by XComponent
    // @param width - NativeWindow width.
    // @param height - NativeWindow height.
    void UpdateNativeWindow(void *window, uint64_t width, uint64_t height);

    void StartAlgorithm(AlgorithmEnum algorithm);
    void StopAlgorithm();
    void UpdateSplitRatio(float ratio);
    void SetScene(int scene);

    // disallow copy and move
    RenderThread(const RenderThread &other) = delete;
    void operator=(const RenderThread &other) = delete;
    RenderThread(RenderThread &&other) = delete;
    void operator=(RenderThread &&other) = delete;

    void PostTask(const RenderTask &task);

    uint64_t GetNativeImageSurfaceId() const {
        std::lock_guard<std::mutex> lock(nativeImageSurfaceIdMutex_);
        return nativeImageSurfaceId_;
    }
    OHNativeWindow *GetNativeImageNativeWindow() const {
        std::lock_guard<std::mutex> lock(nativeImageNativeWindowMutex_);
        return nativeImageNativeWindow_;
    }

    OHNativeWindow *GetNativeWindow();

    int nativeWindowWidth_ = 0;
    int nativeWindowHeight_ = 0;

  private:
    void Start();
    void ThreadMainLoop();

    std::atomic<bool> running_{false};
    std::thread thread_;
    std::thread::id threadId_;

    std::shared_ptr<Algorithm> algorithm_;
    std::atomic<bool> algorithm_running_{false};
    volatile float splitRatio_ = 0.5f;

    // 接收系统发送的Vsync信号，用于控制渲染节奏
    bool InitNativeVsync();
    void DestroyNativeVsync();
    OH_NativeVSync *nativeVsync_ = nullptr;
    static void OnVsync(long long timestamp, void *data);
    std::atomic<int> vSyncCnt_{0};
    mutable std::mutex wakeUpMutex_;
    std::condition_variable wakeUpCond_;
    bool wakeUp_ = false;
    mutable std::mutex taskMutex_;
    std::vector<RenderTask> tasks_;

    int scene_ = 0;

    // renderContext 初始化和清理
    bool InitRenderContext();
    void DestroyRenderContext();
    std::unique_ptr<EglRenderContext> renderContext_;

    // 在渲染线程中执行资源的创建与清理
    bool CreateGLResources();
    void CleanGLResources();

    OHNativeWindow *nativeWindow_ = nullptr;
    EGLSurface eglSurface_ = EGL_NO_SURFACE;

    bool CreateNativeImage();
    void DestroyNativeImage();
    void DestroyAlgorithm();
    void DrawImage();
    static void OnNativeImageFrameAvailable(void *data);
    OH_OnFrameAvailableListener nativeImageFrameAvailableListener_{};
    OH_NativeImage *nativeImage_ = nullptr;
    GLuint nativeImageTexId_ = 0U;
    mutable std::mutex nativeImageSurfaceIdMutex_;
    uint64_t nativeImageSurfaceId_ = 0;
    mutable std::mutex nativeImageNativeWindowMutex_;
    OHNativeWindow *nativeImageNativeWindow_;
    std::atomic<int> availableFrameCnt_{0};

    std::unique_ptr<OesTo2dRenderer> oesTo2dRenderer_;
    std::unique_ptr<SplitScreenRenderer> splitScreenRenderer_;

    GLuint out_texture_2d_ = GL_NONE;
};
} // namespace bmf_lite_demo
#endif // NATIVEIMAGEDEMO_RENDER_THREAD_H
