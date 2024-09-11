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
#ifndef BMFLITE_PLUGIN_RENDER_H
#define BMFLITE_PLUGIN_RENDER_H

#include <ace/xcomponent/native_interface_xcomponent.h>
#include <napi/native_api.h>
#include <string>
#include <unordered_map>

#include "render_thread.h"

namespace bmf_lite_demo {

class PluginRender {
  public:
    explicit PluginRender(std::string &id);
    ~PluginRender() { Release(id_); }
    static PluginRender *GetInstance(std::string &id);
    static void Release(std::string &id);
    static napi_value createCamera(napi_env env, napi_callback_info info);
    static napi_value startCamera(napi_env env, napi_callback_info info);
    static napi_value stopCamera(napi_env env, napi_callback_info info);
    static napi_value releaseCamera(napi_env env, napi_callback_info info);
    static napi_value createPlayer(napi_env env, napi_callback_info info);
    static napi_value setFdSource(napi_env env, napi_callback_info info);
    static napi_value startPlayer(napi_env env, napi_callback_info info);
    static napi_value stopPlayer(napi_env env, napi_callback_info info);
    static napi_value releasePlayer(napi_env env, napi_callback_info info);
    static napi_value startAlgorithm(napi_env env, napi_callback_info info);
    static napi_value stopAlgorithm(napi_env env, napi_callback_info info);

    void Export(napi_env env, napi_value exports);
    void OnSurfaceChanged(OH_NativeXComponent *component, void *window);
    void RegisterCallback(OH_NativeXComponent *nativeXComponent);
    void OnTouchEvent(OH_NativeXComponent *component, void *window);

    void UpdateNativeWindow(void *window, uint64_t width, uint64_t height);
    void CreateCamera();
    void CreatePlayer();

    uint64_t getRenderSurfaceId() {
        return renderThread_->GetNativeImageSurfaceId();
    }

  public:
    static std::unordered_map<std::string, PluginRender *> instance_;
    std::string id_;

  private:
    ;
    OH_NativeXComponent_Callback renderCallback_;
    OH_NativeXComponent_MouseEvent_Callback mouseCallback_;

    std::unique_ptr<RenderThread> renderThread_;
};
} // namespace bmf_lite_demo

#endif // BMFLITE_PLUGIN_RENDER_H
