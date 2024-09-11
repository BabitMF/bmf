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
#include "plugin_render.h"

#include <cstdint>
#include <hilog/log.h>
#include <js_native_api.h>
#include <js_native_api_types.h>
#include <multimedia/player_framework/native_averrors.h>
#include <string>

#include "../common/common.h"
#include "camera/camera_manager.h"
#include "player/player_manager.h"

namespace bmf_lite_demo {
namespace {
static std::unique_ptr<bmf_lite_demo::NDKCamera> ndkCamera_;
static std::unique_ptr<bmf_lite_demo::NDKPlayer> ndkPlayer_;

void OnSurfaceCreatedCB(OH_NativeXComponent *component, void *window) {
    OH_LOG_Print(LOG_APP, LOG_INFO, LOG_PRINT_DOMAIN, "Callback",
                 "OnSurfaceCreatedCB");
    if ((component == nullptr) || (window == nullptr)) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, LOG_PRINT_DOMAIN, "Callback",
                     "OnSurfaceCreatedCB: component or window is null");
        return;
    }

    char idStr[OH_XCOMPONENT_ID_LEN_MAX + 1] = {'\0'};
    uint64_t idSize = OH_XCOMPONENT_ID_LEN_MAX + 1;
    if (OH_NativeXComponent_GetXComponentId(component, idStr, &idSize) !=
        OH_NATIVEXCOMPONENT_RESULT_SUCCESS) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, LOG_PRINT_DOMAIN, "Callback",
                     "OnSurfaceCreatedCB: Unable to get XComponent id");
        return;
    }

    std::string id(idStr);
    auto render = PluginRender::GetInstance(id);
    uint64_t width;
    uint64_t height;
    int32_t xSize = OH_NativeXComponent_GetXComponentSize(component, window,
                                                          &width, &height);
    if ((xSize != OH_NATIVEXCOMPONENT_RESULT_SUCCESS) || (render == nullptr)) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, LOG_PRINT_DOMAIN, "Callback",
                     "OnSurfaceCreatedCB: Unable to get XComponent size");
        return;
    }
    if (render != nullptr) {
        render->UpdateNativeWindow(window, width, height);
    }
}

void OnSurfaceChangedCB(OH_NativeXComponent *component, void *window) {
    OH_LOG_Print(LOG_APP, LOG_INFO, LOG_PRINT_DOMAIN, "Callback",
                 "OnSurfaceChangedCB");
    if ((component == nullptr) || (window == nullptr)) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, LOG_PRINT_DOMAIN, "Callback",
                     "OnSurfaceChangedCB: component or window is null");
        return;
    }

    char idStr[OH_XCOMPONENT_ID_LEN_MAX + 1] = {'\0'};
    uint64_t idSize = OH_XCOMPONENT_ID_LEN_MAX + 1;
    if (OH_NativeXComponent_GetXComponentId(component, idStr, &idSize) !=
        OH_NATIVEXCOMPONENT_RESULT_SUCCESS) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, LOG_PRINT_DOMAIN, "Callback",
                     "OnSurfaceChangedCB: Unable to get XComponent id");
        return;
    }

    std::string id(idStr);
    auto render = PluginRender::GetInstance(id);
    if (render != nullptr) {
        render->OnSurfaceChanged(component, window);
        OH_LOG_Print(LOG_APP, LOG_INFO, LOG_PRINT_DOMAIN, "Callback",
                     "surface changed");
    }
}

void OnSurfaceDestroyedCB(OH_NativeXComponent *component, void *window) {
    OH_LOG_Print(LOG_APP, LOG_INFO, LOG_PRINT_DOMAIN, "Callback",
                 "OnSurfaceDestroyedCB");
    if ((component == nullptr) || (window == nullptr)) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, LOG_PRINT_DOMAIN, "Callback",
                     "OnSurfaceDestroyedCB: component or window is null");
        return;
    }

    char idStr[OH_XCOMPONENT_ID_LEN_MAX + 1] = {'\0'};
    uint64_t idSize = OH_XCOMPONENT_ID_LEN_MAX + 1;
    if (OH_NativeXComponent_GetXComponentId(component, idStr, &idSize) !=
        OH_NATIVEXCOMPONENT_RESULT_SUCCESS) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, LOG_PRINT_DOMAIN, "Callback",
                     "OnSurfaceDestroyedCB: Unable to get XComponent id");
        return;
    }

    std::string id(idStr);
    PluginRender::Release(id);
}

void DispatchTouchEventCB(OH_NativeXComponent *component, void *window) {
    OH_LOG_Print(LOG_APP, LOG_INFO, LOG_PRINT_DOMAIN, "Callback",
                 "DispatchTouchEventCB");
    if ((component == nullptr) || (window == nullptr)) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, LOG_PRINT_DOMAIN, "Callback",
                     "DispatchTouchEventCB: component or window is null");
        return;
    }

    char idStr[OH_XCOMPONENT_ID_LEN_MAX + 1] = {'\0'};
    uint64_t idSize = OH_XCOMPONENT_ID_LEN_MAX + 1;
    if (OH_NativeXComponent_GetXComponentId(component, idStr, &idSize) !=
        OH_NATIVEXCOMPONENT_RESULT_SUCCESS) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, LOG_PRINT_DOMAIN, "Callback",
                     "DispatchTouchEventCB: Unable to get XComponent id");
        return;
    }

    std::string id(idStr);
    PluginRender *render = PluginRender::GetInstance(id);
    if (render != nullptr) {
        render->OnTouchEvent(component, window);
    }
}
} // namespace

std::unordered_map<std::string, PluginRender *> PluginRender::instance_;

PluginRender::PluginRender(std::string &id)
    : renderThread_(std::make_unique<RenderThread>()) {
    this->id_ = id;
}

PluginRender *PluginRender::GetInstance(std::string &id) {
    if (instance_.find(id) == instance_.end()) {
        PluginRender *instance = new PluginRender(id);
        instance_[id] = instance;
        return instance;
    } else {
        return instance_[id];
    }
}

void PluginRender::Export(napi_env env, napi_value exports) {
    if ((env == nullptr) || (exports == nullptr)) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, LOG_PRINT_DOMAIN, "PluginRender",
                     "Export: env or exports is null");
        return;
    }

    napi_property_descriptor desc[] = {
        {"createCamera", nullptr, PluginRender::createCamera, nullptr, nullptr,
         nullptr, napi_default, nullptr},
        {"startCamera", nullptr, PluginRender::startCamera, nullptr, nullptr,
         nullptr, napi_default, nullptr},
        {"stopCamera", nullptr, PluginRender::stopCamera, nullptr, nullptr,
         nullptr, napi_default, nullptr},
        {"releaseCamera", nullptr, PluginRender::releaseCamera, nullptr,
         nullptr, nullptr, napi_default, nullptr},
        {"createPlayer", nullptr, PluginRender::createPlayer, nullptr, nullptr,
         nullptr, napi_default, nullptr},
        {"setFdSource", nullptr, PluginRender::setFdSource, nullptr, nullptr,
         nullptr, napi_default, nullptr},
        {"startPlayer", nullptr, PluginRender::startPlayer, nullptr, nullptr,
         nullptr, napi_default, nullptr},
        {"stopPlayer", nullptr, PluginRender::stopPlayer, nullptr, nullptr,
         nullptr, napi_default, nullptr},
        {"releasePlayer", nullptr, PluginRender::releasePlayer, nullptr,
         nullptr, nullptr, napi_default, nullptr},
        {"startAlgorithm", nullptr, PluginRender::startAlgorithm, nullptr,
         nullptr, nullptr, napi_default, nullptr},
        {"stopAlgorithm", nullptr, PluginRender::stopAlgorithm, nullptr,
         nullptr, nullptr, napi_default, nullptr},
    };
    if (napi_define_properties(env, exports, sizeof(desc) / sizeof(desc[0]),
                               desc) != napi_ok) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, LOG_PRINT_DOMAIN, "PluginRender",
                     "Export: napi_define_properties failed");
    }
}

void PluginRender::Release(std::string &id) {
    PluginRender *render = PluginRender::GetInstance(id);
    if (render != nullptr) {
        instance_.erase(instance_.find(id));
    }
    render->renderThread_ = nullptr;
    //     delete render;
}

napi_value PluginRender::createCamera(napi_env env, napi_callback_info info) {
    napi_value thisArg;
    if (napi_get_cb_info(env, info, nullptr, nullptr, &thisArg, nullptr) !=
        napi_ok) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, LOG_PRINT_DOMAIN, "PluginRender",
                     "createCamera: napi_get_cb_info fail");
        return nullptr;
    }
    /*   // 获取环境变量中XComponent实例*/
    napi_value exportInstance;
    if (napi_get_named_property(env, thisArg, OH_NATIVE_XCOMPONENT_OBJ,
                                &exportInstance) != napi_ok) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, LOG_PRINT_DOMAIN, "PluginRender",
                     "NapiDrawPattern: napi_get_named_property fail");
        return nullptr;
    }

    OH_NativeXComponent *nativeXComponent = nullptr;
    if (napi_unwrap(env, exportInstance,
                    reinterpret_cast<void **>(&nativeXComponent)) != napi_ok) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, LOG_PRINT_DOMAIN, "PluginRender",
                     "NapiDrawPattern: napi_unwrap fail");
        return nullptr;
    }
    // 获取XComponent实例的id
    char idStr[OH_XCOMPONENT_ID_LEN_MAX + 1] = {'\0'};
    uint64_t idSize = OH_XCOMPONENT_ID_LEN_MAX + 1;
    if (OH_NativeXComponent_GetXComponentId(nativeXComponent, idStr, &idSize) !=
        OH_NATIVEXCOMPONENT_RESULT_SUCCESS) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, LOG_PRINT_DOMAIN, "PluginRender",
                     "NapiDrawPattern: Unable to get XComponent id");
        return nullptr;
    }

    std::string id(idStr);
    PluginRender *render = PluginRender::GetInstance(id);
    if (render != nullptr) {
        // 等待renderThread把nativeImage创建成功，这里用这种方式并不是很合适
        while (render->renderThread_->GetNativeImageSurfaceId() == 0) {
            usleep(10000);
        }
        render->CreateCamera();
    }
    return nullptr;
}

napi_value PluginRender::startCamera(napi_env env, napi_callback_info info) {
    OH_LOG_Print(LOG_APP, LOG_INFO, LOG_PRINT_DOMAIN, "PluginRender",
                 "startCamera");
    if (ndkCamera_) {
        ndkCamera_->SessionStart();
    }
    return nullptr;
}

napi_value PluginRender::stopCamera(napi_env env, napi_callback_info info) {
    OH_LOG_Print(LOG_APP, LOG_INFO, LOG_PRINT_DOMAIN, "PluginRender",
                 "stopCamera");
    if (ndkCamera_) {
        ndkCamera_->SessionStop();
    }
    return nullptr;
}

napi_value PluginRender::releaseCamera(napi_env env, napi_callback_info info) {
    OH_LOG_Print(LOG_APP, LOG_INFO, LOG_PRINT_DOMAIN, "PluginRender",
                 "releaseCamera");
    if (ndkCamera_) {
        ndkCamera_->ReleaseCamera();
    }
    ndkCamera_ = nullptr;
    return nullptr;
}

napi_value PluginRender::createPlayer(napi_env env, napi_callback_info info) {
    OH_LOG_Print(LOG_APP, LOG_INFO, LOG_PRINT_DOMAIN, "PluginRender",
                 "createPlayer");
    napi_value thisArg;
    if (napi_get_cb_info(env, info, nullptr, nullptr, &thisArg, nullptr) !=
        napi_ok) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, LOG_PRINT_DOMAIN, "PluginRender",
                     "createCamera: napi_get_cb_info fail");
        return nullptr;
    }
    /*   // 获取环境变量中XComponent实例*/
    napi_value exportInstance;
    if (napi_get_named_property(env, thisArg, OH_NATIVE_XCOMPONENT_OBJ,
                                &exportInstance) != napi_ok) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, LOG_PRINT_DOMAIN, "PluginRender",
                     "NapiDrawPattern: napi_get_named_property fail");
        return nullptr;
    }

    OH_NativeXComponent *nativeXComponent = nullptr;
    if (napi_unwrap(env, exportInstance,
                    reinterpret_cast<void **>(&nativeXComponent)) != napi_ok) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, LOG_PRINT_DOMAIN, "PluginRender",
                     "NapiDrawPattern: napi_unwrap fail");
        return nullptr;
    }
    // 获取XComponent实例的id
    char idStr[OH_XCOMPONENT_ID_LEN_MAX + 1] = {'\0'};
    uint64_t idSize = OH_XCOMPONENT_ID_LEN_MAX + 1;
    if (OH_NativeXComponent_GetXComponentId(nativeXComponent, idStr, &idSize) !=
        OH_NATIVEXCOMPONENT_RESULT_SUCCESS) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, LOG_PRINT_DOMAIN, "PluginRender",
                     "NapiDrawPattern: Unable to get XComponent id");
        return nullptr;
    }

    std::string id(idStr);
    PluginRender *render = PluginRender::GetInstance(id);
    if (render != nullptr) {
        while (render->renderThread_->GetNativeImageNativeWindow() == nullptr) {
            usleep(10000);
        }
        render->CreatePlayer();
    }
    return nullptr;
}

napi_value PluginRender::setFdSource(napi_env env, napi_callback_info info) {
    OH_LOG_Print(LOG_APP, LOG_INFO, LOG_PRINT_DOMAIN, "PluginRender",
                 "setFdSource");
    size_t argc = 3;
    napi_value args[3] = {nullptr};
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);

    napi_valuetype valuetype0;
    napi_typeof(env, args[0], &valuetype0);

    napi_valuetype valuetype1;
    napi_typeof(env, args[1], &valuetype1);

    napi_valuetype valuetype2;
    napi_typeof(env, args[2], &valuetype2);

    int32_t fd;
    napi_get_value_int32(env, args[0], &fd);

    int64_t offset;
    napi_get_value_int64(env, args[1], &offset);

    int64_t size;
    napi_get_value_int64(env, args[2], &size);

    OH_AVErrCode err = OH_AVErrCode::AV_ERR_UNKNOWN;
    if (ndkPlayer_) {
        err = ndkPlayer_->SetFdSource(fd, offset, size);
    }
    if (err == OH_AVErrCode::AV_ERR_OK) {
        OH_LOG_Print(LOG_APP, LOG_INFO, LOG_PRINT_DOMAIN, "PluginRender",
                     "setFdSource success");
    } else {
        OH_LOG_Print(LOG_APP, LOG_ERROR, LOG_PRINT_DOMAIN, "PluginRender",
                     "setFdSource failure, %{public}d", err);
    }
    napi_value output;
    napi_create_uint32(env, err, &output);

    return output;
}

napi_value PluginRender::startPlayer(napi_env env, napi_callback_info info) {
    OH_LOG_Print(LOG_APP, LOG_INFO, LOG_PRINT_DOMAIN, "PluginRender",
                 "startPlayer");
    if (ndkPlayer_) {
        ndkPlayer_->Play();
    }
    return nullptr;
}

napi_value PluginRender::stopPlayer(napi_env env, napi_callback_info info) {
    OH_LOG_Print(LOG_APP, LOG_INFO, LOG_PRINT_DOMAIN, "PluginRender",
                 "stopPlayer");
    if (ndkPlayer_) {
        ndkPlayer_->Stop();
    }
    return nullptr;
}

napi_value PluginRender::releasePlayer(napi_env env, napi_callback_info info) {
    OH_LOG_Print(LOG_APP, LOG_INFO, LOG_PRINT_DOMAIN, "PluginRender",
                 "releasePlayer");
    if (ndkPlayer_) {
        ndkPlayer_->ReleasePlayer();
    }
    ndkPlayer_ = nullptr;
    return nullptr;
}

napi_value PluginRender::startAlgorithm(napi_env env, napi_callback_info info) {
    OH_LOG_Print(LOG_APP, LOG_INFO, LOG_PRINT_DOMAIN, "PluginRender",
                 "startAlgorithm");
    napi_value thisArg;
    if (napi_get_cb_info(env, info, nullptr, nullptr, &thisArg, nullptr) !=
        napi_ok) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, LOG_PRINT_DOMAIN, "PluginRender",
                     "createCamera: napi_get_cb_info fail");
        return nullptr;
    }
    /*   // 获取环境变量中XComponent实例*/
    napi_value exportInstance;
    if (napi_get_named_property(env, thisArg, OH_NATIVE_XCOMPONENT_OBJ,
                                &exportInstance) != napi_ok) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, LOG_PRINT_DOMAIN, "PluginRender",
                     "NapiDrawPattern: napi_get_named_property fail");
        return nullptr;
    }

    OH_NativeXComponent *nativeXComponent = nullptr;
    if (napi_unwrap(env, exportInstance,
                    reinterpret_cast<void **>(&nativeXComponent)) != napi_ok) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, LOG_PRINT_DOMAIN, "PluginRender",
                     "NapiDrawPattern: napi_unwrap fail");
        return nullptr;
    }
    // 获取XComponent实例的id
    char idStr[OH_XCOMPONENT_ID_LEN_MAX + 1] = {'\0'};
    uint64_t idSize = OH_XCOMPONENT_ID_LEN_MAX + 1;
    if (OH_NativeXComponent_GetXComponentId(nativeXComponent, idStr, &idSize) !=
        OH_NATIVEXCOMPONENT_RESULT_SUCCESS) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, LOG_PRINT_DOMAIN, "PluginRender",
                     "NapiDrawPattern: Unable to get XComponent id");
        return nullptr;
    }

    size_t argc = 1;
    napi_value args[1] = {nullptr};
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);

    napi_valuetype valuetype0;
    napi_typeof(env, args[0], &valuetype0);

    int32_t alg_type;
    napi_get_value_int32(env, args[0], &alg_type);

    std::string id(idStr);
    PluginRender *render = PluginRender::GetInstance(id);
    if (render != nullptr) {
        if (alg_type == 0) {
            render->renderThread_->StartAlgorithm(
                AlgorithmEnum::SUPER_RESOLUTION);
        } else if (alg_type == 1) {
            render->renderThread_->StartAlgorithm(AlgorithmEnum::DENOISE);
        }
    }
    return nullptr;
}

napi_value PluginRender::stopAlgorithm(napi_env env, napi_callback_info info) {
    OH_LOG_Print(LOG_APP, LOG_INFO, LOG_PRINT_DOMAIN, "PluginRender",
                 "stopAlgorithm");
    napi_value thisArg;
    if (napi_get_cb_info(env, info, nullptr, nullptr, &thisArg, nullptr) !=
        napi_ok) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, LOG_PRINT_DOMAIN, "PluginRender",
                     "createCamera: napi_get_cb_info fail");
        return nullptr;
    }
    /*   // 获取环境变量中XComponent实例*/
    napi_value exportInstance;
    if (napi_get_named_property(env, thisArg, OH_NATIVE_XCOMPONENT_OBJ,
                                &exportInstance) != napi_ok) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, LOG_PRINT_DOMAIN, "PluginRender",
                     "NapiDrawPattern: napi_get_named_property fail");
        return nullptr;
    }

    OH_NativeXComponent *nativeXComponent = nullptr;
    if (napi_unwrap(env, exportInstance,
                    reinterpret_cast<void **>(&nativeXComponent)) != napi_ok) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, LOG_PRINT_DOMAIN, "PluginRender",
                     "NapiDrawPattern: napi_unwrap fail");
        return nullptr;
    }
    // 获取XComponent实例的id
    char idStr[OH_XCOMPONENT_ID_LEN_MAX + 1] = {'\0'};
    uint64_t idSize = OH_XCOMPONENT_ID_LEN_MAX + 1;
    if (OH_NativeXComponent_GetXComponentId(nativeXComponent, idStr, &idSize) !=
        OH_NATIVEXCOMPONENT_RESULT_SUCCESS) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, LOG_PRINT_DOMAIN, "PluginRender",
                     "NapiDrawPattern: Unable to get XComponent id");
        return nullptr;
    }

    size_t argc = 1;
    napi_value args[1] = {nullptr};
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);

    napi_valuetype valuetype0;
    napi_typeof(env, args[0], &valuetype0);

    int32_t alg_type;
    napi_get_value_int32(env, args[0], &alg_type);

    std::string id(idStr);
    PluginRender *render = PluginRender::GetInstance(id);
    if (render != nullptr) {
        render->renderThread_->StopAlgorithm();
    }
    return nullptr;
}

void PluginRender::CreateCamera() {
    std::string surfaceIdStr =
        std::to_string(renderThread_->GetNativeImageSurfaceId());
    ndkCamera_ = std::make_unique<bmf_lite_demo::NDKCamera>(
        surfaceIdStr.c_str(), FOCUS_MODE_AUTO, 0);
    renderThread_->SetScene(1);
    OH_LOG_Print(LOG_APP, LOG_INFO, LOG_PRINT_DOMAIN, "PluginRender",
                 "Create the Camera");
}

void PluginRender::CreatePlayer() {
    OH_LOG_Print(LOG_APP, LOG_INFO, LOG_PRINT_DOMAIN, "PluginRender",
                 "CreatePlayer nativeWindow_ %{public}p",
                 renderThread_->GetNativeImageNativeWindow());
    ndkPlayer_ = std::make_unique<bmf_lite_demo::NDKPlayer>(
        renderThread_->GetNativeImageNativeWindow());
    renderThread_->SetScene(0);
    OH_LOG_Print(LOG_APP, LOG_INFO, LOG_PRINT_DOMAIN, "PluginRender",
                 "Create the Player");
}

void PluginRender::OnSurfaceChanged(OH_NativeXComponent *component,
                                    void *window) {
    char idStr[OH_XCOMPONENT_ID_LEN_MAX + 1] = {'\0'};
    uint64_t idSize = OH_XCOMPONENT_ID_LEN_MAX + 1;
    if (OH_NativeXComponent_GetXComponentId(component, idStr, &idSize) !=
        OH_NATIVEXCOMPONENT_RESULT_SUCCESS) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, LOG_PRINT_DOMAIN, "Callback",
                     "OnSurfaceChanged: Unable to get XComponent id");
        return;
    }

    std::string id(idStr);
    PluginRender *render = PluginRender::GetInstance(id);
    double offsetX;
    double offsetY;
    OH_NativeXComponent_GetXComponentOffset(component, window, &offsetX,
                                            &offsetY);
    OH_LOG_Print(LOG_APP, LOG_INFO, LOG_PRINT_DOMAIN,
                 "OH_NativeXComponent_GetXComponentOffset",
                 "offsetX = %{public}lf, offsetY = %{public}lf", offsetX,
                 offsetY);
    uint64_t width;
    uint64_t height;
    OH_NativeXComponent_GetXComponentSize(component, window, &width, &height);
    UpdateNativeWindow(window, width, height);
}

void PluginRender::UpdateNativeWindow(void *window, uint64_t width,
                                      uint64_t height) {
    renderThread_->UpdateNativeWindow(window, width, height);
}

void PluginRender::OnTouchEvent(OH_NativeXComponent *component, void *window) {
    OH_NativeXComponent_TouchEvent touchEvent;
    OH_NativeXComponent_GetTouchEvent(component, window, &touchEvent);

    float splitRatio = touchEvent.x / renderThread_->nativeWindowWidth_;
    OH_LOG_Print(LOG_APP, LOG_INFO, LOG_PRINT_DOMAIN, "Callback",
                 "OnTouchEvent touchEvent %{public}f", splitRatio);
    renderThread_->UpdateSplitRatio(splitRatio);
}

void PluginRender::RegisterCallback(OH_NativeXComponent *nativeXComponent) {
    renderCallback_.OnSurfaceCreated = OnSurfaceCreatedCB;
    renderCallback_.OnSurfaceChanged = OnSurfaceChangedCB;
    renderCallback_.OnSurfaceDestroyed = OnSurfaceDestroyedCB;
    renderCallback_.DispatchTouchEvent = DispatchTouchEventCB;
    OH_NativeXComponent_RegisterCallback(nativeXComponent, &renderCallback_);
}

} // namespace bmf_lite_demo