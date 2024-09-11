/*
 * Copyright (c) 2023-2023 Huawei Device Co., Ltd.
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
#include "camera_manager.h"

namespace bmf_lite_demo {
std::mutex NDKCamera::mtx_;

NDKCamera::NDKCamera(const char *surfaceId, uint32_t focusMode,
                     uint32_t cameraDeviceIndex)
    : previewSurfaceId_(surfaceId), cameras_(nullptr), focusMode_(focusMode),
      cameraDeviceIndex_(cameraDeviceIndex), cameraOutputCapability_(nullptr),
      cameraInput_(nullptr), captureSession_(nullptr), size_(0),
      isCameraMuted_(nullptr), profile_(nullptr), photoSurfaceId_(nullptr),
      previewOutput_(nullptr), photoOutput_(nullptr),
      metaDataObjectType_(nullptr), metadataOutput_(nullptr),
      isExposureModeSupported_(false), isFocusModeSupported_(false),
      exposureMode_(EXPOSURE_MODE_LOCKED), minExposureBias_(0),
      maxExposureBias_(0), step_(0), ret_(CAMERA_OK) {
    valid_ = false;
    ReleaseCamera();
    Camera_ErrorCode ret = OH_Camera_GetCameraManager(&cameraManager_);
    if (cameraManager_ == nullptr || ret != CAMERA_OK) {
        OH_LOG_ERROR(LOG_APP, "Get CameraManager failed.");
    }

    ret =
        OH_CameraManager_CreateCaptureSession(cameraManager_, &captureSession_);
    if (captureSession_ == nullptr || ret != CAMERA_OK) {
        OH_LOG_ERROR(LOG_APP, "Create captureSession failed.");
    }
    CaptureSessionRegisterCallback();
    GetSupportedCameras();
    GetSupportedOutputCapability();
    CreatePreviewOutput();
    CreateCameraInput();
    CameraInputOpen();
    CameraManagerRegisterCallback();
    SessionFlowFn();
    valid_ = true;
}

NDKCamera::~NDKCamera() {
    valid_ = false;
    OH_LOG_ERROR(LOG_APP, "~NDKCamera");
    Camera_ErrorCode ret = CAMERA_OK;

    if (cameraManager_) {
        OH_LOG_ERROR(
            LOG_APP,
            "Release OH_CameraManager_DeleteSupportedCameraOutputCapability. "
            "enter");
        ret = OH_CameraManager_DeleteSupportedCameraOutputCapability(
            cameraManager_, cameraOutputCapability_);
        if (ret != CAMERA_OK) {
            OH_LOG_ERROR(LOG_APP, "Delete CameraOutputCapability failed.");
        } else {
            OH_LOG_ERROR(
                LOG_APP,
                "Release "
                "OH_CameraManager_DeleteSupportedCameraOutputCapability. ok");
        }

        OH_LOG_ERROR(LOG_APP,
                     "Release OH_CameraManager_DeleteSupportedCameras. enter");
        ret = OH_CameraManager_DeleteSupportedCameras(cameraManager_, cameras_,
                                                      size_);
        if (ret != CAMERA_OK) {
            OH_LOG_ERROR(LOG_APP, "Delete Cameras failed.");
        } else {
            OH_LOG_ERROR(LOG_APP,
                         "Release OH_CameraManager_DeleteSupportedCameras. ok");
        }
        ret = OH_Camera_DeleteCameraManager(cameraManager_);
        if (ret != CAMERA_OK) {
            OH_LOG_ERROR(LOG_APP, "Delete CameraManager failed.");
        } else {
            OH_LOG_ERROR(LOG_APP, "Release OH_Camera_DeleteCameraMananger. ok");
        }
        cameraManager_ = nullptr;
    }
    OH_LOG_ERROR(LOG_APP, "~NDKCamera exit");
}

Camera_ErrorCode NDKCamera::ReleaseCamera(void) {
    OH_LOG_ERROR(LOG_APP, " enter ReleaseCamera");
    if (previewOutput_) {
        PreviewOutputStop();
        PreviewOutputRelease();
        OH_CaptureSession_RemovePreviewOutput(captureSession_, previewOutput_);
    }
    if (photoOutput_) {
        PhotoOutputRelease();
    }
    if (videoOutput_) {
        VideoOutputStop();
    }
    if (captureSession_) {
        SessionRelease();
    }
    if (cameraInput_) {
        CameraInputClose();
    }
    //     NDKCamera::Destroy();
    OH_LOG_ERROR(LOG_APP, " exit ReleaseCamera");
    return ret_;
}

Camera_ErrorCode NDKCamera::ReleaseSession(void) {
    OH_LOG_ERROR(LOG_APP, " enter ReleaseSession");
    PreviewOutputStop();
    PhotoOutputRelease();
    SessionRelease();
    OH_LOG_ERROR(LOG_APP, " exit ReleaseSession");
    return ret_;
}

Camera_ErrorCode NDKCamera::SessionRelease(void) {
    OH_LOG_ERROR(LOG_APP, " enter SessionRealese");
    Camera_ErrorCode ret = OH_CaptureSession_Release(captureSession_);
    captureSession_ = nullptr;
    OH_LOG_ERROR(LOG_APP, " exit SessionRealese");
    return ret;
}

Camera_ErrorCode NDKCamera::HasFlashFn(uint32_t mode) {
    Camera_FlashMode flashMode = static_cast<Camera_FlashMode>(mode);
    // Check for flashing lights
    bool hasFlash = false;
    Camera_ErrorCode ret =
        OH_CaptureSession_HasFlash(captureSession_, &hasFlash);
    if (captureSession_ == nullptr || ret != CAMERA_OK) {
        OH_LOG_ERROR(LOG_APP, "OH_CaptureSession_HasFlash failed.");
    }
    if (hasFlash) {
        OH_LOG_INFO(LOG_APP, "hasFlash success-----");
    } else {
        OH_LOG_ERROR(LOG_APP, "hasFlash fail-----");
    }

    // Check if the flash mode is supported
    bool isSupported = false;
    ret = OH_CaptureSession_IsFlashModeSupported(captureSession_, flashMode,
                                                 &isSupported);
    if (ret != CAMERA_OK) {
        OH_LOG_ERROR(LOG_APP, "OH_CaptureSession_IsFlashModeSupported failed.");
    }
    if (isSupported) {
        OH_LOG_INFO(LOG_APP, "isFlashModeSupported success-----");
    } else {
        OH_LOG_ERROR(LOG_APP, "isFlashModeSupported fail-----");
    }

    // Set flash mode
    ret = OH_CaptureSession_SetFlashMode(captureSession_, flashMode);
    if (ret == CAMERA_OK) {
        OH_LOG_INFO(LOG_APP, "OH_CaptureSession_SetFlashMode success.");
    } else {
        OH_LOG_ERROR(LOG_APP,
                     "OH_CaptureSession_SetFlashMode failed. %{public}d ", ret);
    }

    // Obtain the flash mode of the current device
    ret = OH_CaptureSession_GetFlashMode(captureSession_, &flashMode);
    if (ret == CAMERA_OK) {
        OH_LOG_INFO(
            LOG_APP,
            "OH_CaptureSession_GetFlashMode success. flashMode：%{public}d ",
            flashMode);
    } else {
        OH_LOG_ERROR(LOG_APP, "OH_CaptureSession_GetFlashMode failed. %d ",
                     ret);
    }
    return ret;
}

Camera_ErrorCode NDKCamera::IsVideoStabilizationModeSupportedFn(uint32_t mode) {
    Camera_VideoStabilizationMode videoMode =
        static_cast<Camera_VideoStabilizationMode>(mode);
    // Check if the specified video anti shake mode is supported
    bool isSupported = false;
    Camera_ErrorCode ret = OH_CaptureSession_IsVideoStabilizationModeSupported(
        captureSession_, videoMode, &isSupported);
    if (ret != CAMERA_OK) {
        OH_LOG_ERROR(
            LOG_APP,
            "OH_CaptureSession_IsVideoStabilizationModeSupported failed.");
    }
    if (isSupported) {
        OH_LOG_INFO(
            LOG_APP,
            "OH_CaptureSession_IsVideoStabilizationModeSupported success-----");
    } else {
        OH_LOG_ERROR(
            LOG_APP,
            "OH_CaptureSession_IsVideoStabilizationModeSupported fail-----");
    }

    // Set video stabilization
    ret =
        OH_CaptureSession_SetVideoStabilizationMode(captureSession_, videoMode);
    if (ret == CAMERA_OK) {
        OH_LOG_INFO(LOG_APP,
                    "OH_CaptureSession_SetVideoStabilizationMode success.");
    } else {
        OH_LOG_ERROR(
            LOG_APP,
            "OH_CaptureSession_SetVideoStabilizationMode failed. %{public}d ",
            ret);
    }

    ret = OH_CaptureSession_GetVideoStabilizationMode(captureSession_,
                                                      &videoMode);
    if (ret == CAMERA_OK) {
        OH_LOG_INFO(LOG_APP,
                    "OH_CaptureSession_GetVideoStabilizationMode success. "
                    "videoMode：%u ",
                    videoMode);
    } else {
        OH_LOG_ERROR(
            LOG_APP,
            "OH_CaptureSession_GetVideoStabilizationMode failed. %{public}d ",
            ret);
    }
    return ret;
}

Camera_ErrorCode NDKCamera::setZoomRatioFn(uint32_t zoomRatio) {
    float zoom = float(zoomRatio);
    // Obtain supported zoom range
    float minZoom;
    float maxZoom;
    Camera_ErrorCode ret = OH_CaptureSession_GetZoomRatioRange(
        captureSession_, &minZoom, &maxZoom);
    if (captureSession_ == nullptr || ret != CAMERA_OK) {
        OH_LOG_ERROR(LOG_APP, "OH_CaptureSession_GetZoomRatioRange failed.");
    } else {
        OH_LOG_INFO(LOG_APP,
                    "OH_CaptureSession_GetZoomRatioRange success. minZoom: "
                    "%{public}f, maxZoom:%{public}f",
                    minZoom, maxZoom);
    }

    // Set Zoom
    ret = OH_CaptureSession_SetZoomRatio(captureSession_, zoom);
    if (ret == CAMERA_OK) {
        OH_LOG_INFO(LOG_APP, "OH_CaptureSession_SetZoomRatio success.");
    } else {
        OH_LOG_ERROR(LOG_APP,
                     "OH_CaptureSession_SetZoomRatio failed. %{public}d ", ret);
    }

    // Obtain the zoom value of the current device
    ret = OH_CaptureSession_GetZoomRatio(captureSession_, &zoom);
    if (ret == CAMERA_OK) {
        OH_LOG_INFO(LOG_APP,
                    "OH_CaptureSession_GetZoomRatio success. zoom：%{public}f ",
                    zoom);
    } else {
        OH_LOG_ERROR(LOG_APP,
                     "OH_CaptureSession_GetZoomRatio failed. %{public}d ", ret);
    }
    return ret;
}

Camera_ErrorCode NDKCamera::SessionBegin(void) {
    Camera_ErrorCode ret = OH_CaptureSession_BeginConfig(captureSession_);
    if (ret == CAMERA_OK) {
        OH_LOG_INFO(LOG_APP, "OH_CaptureSession_BeginConfig success.");
    } else {
        OH_LOG_ERROR(LOG_APP, "OH_CaptureSession_BeginConfig failed. %d ", ret);
    }
    return ret;
}

Camera_ErrorCode NDKCamera::SessionCommitConfig(void) {
    Camera_ErrorCode ret = OH_CaptureSession_CommitConfig(captureSession_);
    if (ret == CAMERA_OK) {
        OH_LOG_INFO(LOG_APP, "OH_CaptureSession_CommitConfig success.");
    } else {
        OH_LOG_ERROR(LOG_APP, "OH_CaptureSession_CommitConfig failed. %d ",
                     ret);
    }
    return ret;
}

Camera_ErrorCode NDKCamera::SessionStart(void) {
    Camera_ErrorCode ret = OH_CaptureSession_Start(captureSession_);
    if (ret == CAMERA_OK) {
        OH_LOG_INFO(LOG_APP, "OH_CaptureSession_Start success.");
    } else {
        OH_LOG_ERROR(LOG_APP, "OH_CaptureSession_Start failed. %d ", ret);
    }
    return ret;
}

Camera_ErrorCode NDKCamera::SessionStop(void) {
    Camera_ErrorCode ret = OH_CaptureSession_Stop(captureSession_);
    if (ret == CAMERA_OK) {
        OH_LOG_INFO(LOG_APP, "OH_CaptureSession_Stop success.");
    } else {
        OH_LOG_ERROR(LOG_APP, "OH_CaptureSession_Stop failed. %d ", ret);
    }
    return ret;
}

Camera_ErrorCode NDKCamera::SessionFlowFn(void) {
    OH_LOG_INFO(LOG_APP, "Start SessionFlowFn IN.");
    // Start configuring session
    OH_LOG_INFO(LOG_APP, "session beginConfig.");
    Camera_ErrorCode ret = OH_CaptureSession_BeginConfig(captureSession_);

    // Add CameraInput to the session
    OH_LOG_INFO(LOG_APP, "session addInput.");
    ret = OH_CaptureSession_AddInput(captureSession_, cameraInput_);

    // Add previewOutput to the session
    OH_LOG_INFO(LOG_APP, "session add Preview Output.");
    ret = OH_CaptureSession_AddPreviewOutput(captureSession_, previewOutput_);

    // Adding PhotoOutput to the Session
    OH_LOG_INFO(LOG_APP, "session add Photo Output.");

    // Submit configuration information
    OH_LOG_INFO(LOG_APP, "session commitConfig");
    ret = OH_CaptureSession_CommitConfig(captureSession_);

    // Start Session Work
    OH_LOG_INFO(LOG_APP, "session start");
    ret = OH_CaptureSession_Start(captureSession_);
    OH_LOG_INFO(LOG_APP, "session success");

    // Start focusing
    OH_LOG_INFO(LOG_APP, "IsFocusMode start");
    ret = IsFocusMode(focusMode_);
    OH_LOG_INFO(LOG_APP, "IsFocusMode success");
    return ret;
}

Camera_ErrorCode NDKCamera::CreateCameraInput(void) {
    OH_LOG_ERROR(LOG_APP, "enter CreateCameraInput.");
    ret_ = OH_CameraManager_CreateCameraInput(
        cameraManager_, &cameras_[cameraDeviceIndex_], &cameraInput_);
    if (cameraInput_ == nullptr || ret_ != CAMERA_OK) {
        OH_LOG_ERROR(LOG_APP, "CreateCameraInput failed.");
        return CAMERA_INVALID_ARGUMENT;
    }
    OH_LOG_ERROR(LOG_APP, "exit CreateCameraInput.");
    CameraInputRegisterCallback();
    return ret_;
}

Camera_ErrorCode NDKCamera::CameraInputOpen(void) {
    OH_LOG_ERROR(LOG_APP, "enter CameraInputOpen.");
    ret_ = OH_CameraInput_Open(cameraInput_);
    if (ret_ != CAMERA_OK) {
        OH_LOG_ERROR(LOG_APP, "CameraInput_Open failed.");
        return CAMERA_INVALID_ARGUMENT;
    }
    OH_LOG_ERROR(LOG_APP, "exit CameraInputOpen.");
    return ret_;
}

Camera_ErrorCode NDKCamera::CameraInputClose(void) {
    OH_LOG_ERROR(LOG_APP, "enter CameraInput_Close.");
    ret_ = OH_CameraInput_Close(cameraInput_);
    if (ret_ != CAMERA_OK) {
        OH_LOG_ERROR(LOG_APP, "CameraInput_Close failed.");
        return CAMERA_INVALID_ARGUMENT;
    }
    OH_LOG_ERROR(LOG_APP, "exit CameraInput_Close.");
    return ret_;
}

Camera_ErrorCode NDKCamera::CameraInputRelease(void) {
    OH_LOG_ERROR(LOG_APP, "enter CameraInputRelease.");
    ret_ = OH_CameraInput_Release(cameraInput_);
    if (ret_ != CAMERA_OK) {
        OH_LOG_ERROR(LOG_APP, "CameraInput_Release failed.");
        return CAMERA_INVALID_ARGUMENT;
    }
    OH_LOG_ERROR(LOG_APP, "exit CameraInputRelease.");
    return ret_;
}

Camera_ErrorCode NDKCamera::GetSupportedCameras(void) {
    ret_ =
        OH_CameraManager_GetSupportedCameras(cameraManager_, &cameras_, &size_);
    if (cameras_ == nullptr || ret_ != CAMERA_OK) {
        OH_LOG_ERROR(LOG_APP, "Get supported cameras failed.");
        return CAMERA_INVALID_ARGUMENT;
    }
    return ret_;
}

Camera_ErrorCode NDKCamera::GetSupportedOutputCapability(void) {
    ret_ = OH_CameraManager_GetSupportedCameraOutputCapability(
        cameraManager_, &cameras_[cameraDeviceIndex_],
        &cameraOutputCapability_);
    if (cameraOutputCapability_ == nullptr || ret_ != CAMERA_OK) {
        OH_LOG_ERROR(LOG_APP, "GetSupportedCameraOutputCapability failed.");
        return CAMERA_INVALID_ARGUMENT;
    }
    return ret_;
}

Camera_ErrorCode NDKCamera::CreatePreviewOutput(void) {
    for (int i = 0; i < cameraOutputCapability_->previewProfilesSize; i++) {
        if (cameraOutputCapability_->previewProfiles[i]->size.height == 720 &&
            cameraOutputCapability_->previewProfiles[i]->size.width == 1280) {
            profile_ = cameraOutputCapability_->previewProfiles[i];
            break;
        }
    }
    //    profile_ = cameraOutputCapability_->previewProfiles[0];
    OH_LOG_ERROR(LOG_APP, "Get CreatePreviewOutput %d %d", profile_->size.width,
                 profile_->size.height);
    if (profile_ == nullptr) {
        OH_LOG_ERROR(LOG_APP, "Get previewProfiles failed.");
        return CAMERA_INVALID_ARGUMENT;
    }
    ret_ = OH_CameraManager_CreatePreviewOutput(
        cameraManager_, profile_, previewSurfaceId_, &previewOutput_);
    if (previewSurfaceId_ == nullptr || previewOutput_ == nullptr ||
        ret_ != CAMERA_OK) {
        OH_LOG_ERROR(LOG_APP, "CreatePreviewOutput failed.");
        return CAMERA_INVALID_ARGUMENT;
    }
    PreviewOutputRegisterCallback();
    return ret_;
}

Camera_ErrorCode NDKCamera::CreatePhotoOutput(char *photoSurfaceId) {
    profile_ = cameraOutputCapability_->photoProfiles[0];
    if (profile_ == nullptr) {
        OH_LOG_ERROR(LOG_APP, "Get photoProfiles failed.");
        return CAMERA_INVALID_ARGUMENT;
    }

    if (photoSurfaceId == nullptr) {
        OH_LOG_ERROR(LOG_APP, "CreatePhotoOutput failed.");
        return CAMERA_INVALID_ARGUMENT;
    }

    ret_ = OH_CameraManager_CreatePhotoOutput(cameraManager_, profile_,
                                              photoSurfaceId, &photoOutput_);
    PhotoOutputRegisterCallback();
    return ret_;
}

Camera_ErrorCode NDKCamera::CreateVideoOutput(const char *videoId) {
    videoProfile_ = cameraOutputCapability_->videoProfiles[0];

    if (videoProfile_ == nullptr) {
        OH_LOG_ERROR(LOG_APP, "Get videoProfiles failed.");
        return CAMERA_INVALID_ARGUMENT;
    }
    ret_ = OH_CameraManager_CreateVideoOutput(cameraManager_, videoProfile_,
                                              videoId, &videoOutput_);
    if (videoId == nullptr || videoOutput_ == nullptr || ret_ != CAMERA_OK) {
        OH_LOG_ERROR(LOG_APP, "CreateVideoOutput failed.");
        return CAMERA_INVALID_ARGUMENT;
    }

    return ret_;
}

Camera_ErrorCode NDKCamera::AddVideoOutput(void) {
    Camera_ErrorCode ret =
        OH_CaptureSession_AddVideoOutput(captureSession_, videoOutput_);
    if (ret == CAMERA_OK) {
        OH_LOG_ERROR(LOG_APP, "OH_CaptureSession_AddVideoOutput success.");
    } else {
        OH_LOG_ERROR(LOG_APP, "OH_CaptureSession_AddVideoOutput failed. %d ",
                     ret);
    }
    return ret;
}

Camera_ErrorCode NDKCamera::AddPhotoOutput() {
    Camera_ErrorCode ret =
        OH_CaptureSession_AddPhotoOutput(captureSession_, photoOutput_);
    if (ret == CAMERA_OK) {
        OH_LOG_ERROR(LOG_APP, "OH_CaptureSession_AddPhotoOutput success.");
    } else {
        OH_LOG_ERROR(LOG_APP, "OH_CaptureSession_AddPhotoOutput failed. %d ",
                     ret);
    }
    return ret;
}

Camera_ErrorCode NDKCamera::CreateMetadataOutput(void) {
    metaDataObjectType_ =
        cameraOutputCapability_->supportedMetadataObjectTypes[0];
    if (metaDataObjectType_ == nullptr) {
        OH_LOG_ERROR(LOG_APP, "Get metaDataObjectType failed.");
        return CAMERA_INVALID_ARGUMENT;
    }
    ret_ = OH_CameraManager_CreateMetadataOutput(
        cameraManager_, metaDataObjectType_, &metadataOutput_);
    if (metadataOutput_ == nullptr || ret_ != CAMERA_OK) {
        OH_LOG_ERROR(LOG_APP, "CreateMetadataOutput failed.");
        return CAMERA_INVALID_ARGUMENT;
    }
    MetadataOutputRegisterCallback();
    return ret_;
}

Camera_ErrorCode NDKCamera::IsCameraMuted(void) {
    ret_ = OH_CameraManager_IsCameraMuted(cameraManager_, isCameraMuted_);
    if (isCameraMuted_ == nullptr || ret_ != CAMERA_OK) {
        OH_LOG_ERROR(LOG_APP, "IsCameraMuted failed.");
        return CAMERA_INVALID_ARGUMENT;
    }
    return ret_;
}

Camera_ErrorCode NDKCamera::PreviewOutputStop(void) {
    OH_LOG_ERROR(LOG_APP, "enter PreviewOutputStop.");
    ret_ = OH_PreviewOutput_Stop(previewOutput_);
    if (ret_ != CAMERA_OK) {
        OH_LOG_ERROR(LOG_APP, "PreviewOutputStop failed.");
        return CAMERA_INVALID_ARGUMENT;
    }
    return ret_;
}

Camera_ErrorCode NDKCamera::PreviewOutputRelease(void) {
    OH_LOG_ERROR(LOG_APP, "enter PreviewOutputRelease.");
    ret_ = OH_PreviewOutput_Release(previewOutput_);
    if (ret_ != CAMERA_OK) {
        OH_LOG_ERROR(LOG_APP, "PreviewOutputRelease failed.");
        return CAMERA_INVALID_ARGUMENT;
    }
    return ret_;
}

Camera_ErrorCode NDKCamera::PhotoOutputRelease(void) {
    OH_LOG_ERROR(LOG_APP, "enter PhotoOutputRelease.");
    ret_ = OH_PhotoOutput_Release(photoOutput_);
    if (ret_ != CAMERA_OK) {
        OH_LOG_ERROR(LOG_APP, "PhotoOutputRelease failed.");
        return CAMERA_INVALID_ARGUMENT;
    }
    return ret_;
}

Camera_ErrorCode NDKCamera::StartVideo(char *videoId, char *photoId) {
    OH_LOG_INFO(LOG_APP, "StartVideo begin.");
    Camera_ErrorCode ret = SessionStop();
    if (ret == CAMERA_OK) {
        OH_LOG_INFO(LOG_APP, "SessionStop success.");
    } else {
        OH_LOG_ERROR(LOG_APP, "SessionStop failed. %d ", ret);
    }
    ret = SessionBegin();
    if (ret == CAMERA_OK) {
        OH_LOG_INFO(LOG_APP, "SessionBegin success.");
    } else {
        OH_LOG_ERROR(LOG_APP, "SessionBegin failed. %d ", ret);
    }
    OH_CaptureSession_RemovePhotoOutput(captureSession_, photoOutput_);
    CreatePhotoOutput(photoId);
    AddPhotoOutput();
    CreateVideoOutput(videoId);
    AddVideoOutput();
    SessionCommitConfig();
    SessionStart();
    VideoOutputRegisterCallback();
    return ret;
}

Camera_ErrorCode NDKCamera::VideoOutputStart(void) {
    OH_LOG_INFO(LOG_APP, "VideoOutputStart begin.");
    Camera_ErrorCode ret = OH_VideoOutput_Start(videoOutput_);
    if (ret == CAMERA_OK) {
        OH_LOG_INFO(LOG_APP, "OH_VideoOutput_Start success.");
    } else {
        OH_LOG_ERROR(LOG_APP, "OH_VideoOutput_Start failed. %d ", ret);
    }
    return ret;
}

Camera_ErrorCode NDKCamera::StartPhoto(char *mSurfaceId) {
    Camera_ErrorCode ret = CAMERA_OK;
    if (takePictureTimes == 0) {
        ret = SessionStop();
        if (ret == CAMERA_OK) {
            OH_LOG_INFO(LOG_APP, "SessionStop success.");
        } else {
            OH_LOG_ERROR(LOG_APP, "SessionStop failed. %d ", ret);
        }
        ret = SessionBegin();
        if (ret == CAMERA_OK) {
            OH_LOG_INFO(LOG_APP, "SessionBegin success.");
        } else {
            OH_LOG_ERROR(LOG_APP, "SessionBegin failed. %d ", ret);
        }
        OH_LOG_INFO(LOG_APP, "startPhoto begin.");
        ret = CreatePhotoOutput(mSurfaceId);

        OH_LOG_INFO(LOG_APP, "startPhoto CreatePhotoOutput ret = %{public}d.",
                    ret);
        ret = OH_CaptureSession_AddPhotoOutput(captureSession_, photoOutput_);
        OH_LOG_INFO(LOG_APP, "startPhoto AddPhotoOutput ret = %{public}d.",
                    ret);
        ret = SessionCommitConfig();

        OH_LOG_INFO(LOG_APP, "startPhoto SessionCommitConfig ret = %{public}d.",
                    ret);
        ret = SessionStart();
        OH_LOG_INFO(LOG_APP, "startPhoto SessionStart ret = %{public}d.", ret);
    }
    ret = TakePicture();
    OH_LOG_INFO(LOG_APP, "startPhoto OH_PhotoOutput_Capture ret = %{public}d.",
                ret);
    if (ret_ != CAMERA_OK) {
        OH_LOG_ERROR(LOG_APP, "startPhoto failed.");
        return CAMERA_INVALID_ARGUMENT;
    }
    takePictureTimes++;
    return ret_;
}

// exposure mode
Camera_ErrorCode NDKCamera::IsExposureModeSupportedFn(uint32_t mode) {
    OH_LOG_INFO(LOG_APP, "IsExposureModeSupportedFn start.");
    exposureMode_ = static_cast<Camera_ExposureMode>(mode);
    ret_ = OH_CaptureSession_IsExposureModeSupported(
        captureSession_, exposureMode_, &isExposureModeSupported_);
    if (ret_ != CAMERA_OK) {
        OH_LOG_ERROR(LOG_APP, "IsExposureModeSupported failed.");
        return CAMERA_INVALID_ARGUMENT;
    }
    ret_ = OH_CaptureSession_SetExposureMode(captureSession_, exposureMode_);
    if (ret_ != CAMERA_OK) {
        OH_LOG_ERROR(LOG_APP, "SetExposureMode failed.");
        return CAMERA_INVALID_ARGUMENT;
    }
    ret_ = OH_CaptureSession_GetExposureMode(captureSession_, &exposureMode_);
    if (ret_ != CAMERA_OK) {
        OH_LOG_ERROR(LOG_APP, "GetExposureMode failed.");
        return CAMERA_INVALID_ARGUMENT;
    }
    OH_LOG_INFO(LOG_APP, "IsExposureModeSupportedFn end.");
    return ret_;
}

Camera_ErrorCode NDKCamera::IsMeteringPoint(int x, int y) {
    OH_LOG_INFO(LOG_APP, "IsMeteringPoint start.");
    ret_ = OH_CaptureSession_GetExposureMode(captureSession_, &exposureMode_);
    if (ret_ != CAMERA_OK) {
        OH_LOG_ERROR(LOG_APP, "GetExposureMode failed.");
        return CAMERA_INVALID_ARGUMENT;
    }
    Camera_Point exposurePoint;
    exposurePoint.x = x;
    exposurePoint.y = y;
    ret_ = OH_CaptureSession_SetMeteringPoint(captureSession_, exposurePoint);
    if (ret_ != CAMERA_OK) {
        OH_LOG_ERROR(LOG_APP, "SetMeteringPoint failed.");
        return CAMERA_INVALID_ARGUMENT;
    }
    ret_ = OH_CaptureSession_GetMeteringPoint(captureSession_, &exposurePoint);
    if (ret_ != CAMERA_OK) {
        OH_LOG_ERROR(LOG_APP, "GetMeteringPoint failed.");
        return CAMERA_INVALID_ARGUMENT;
    }
    OH_LOG_INFO(LOG_APP, "IsMeteringPoint end.");
    return ret_;
}

Camera_ErrorCode NDKCamera::IsExposureBiasRange(int exposureBias) {
    OH_LOG_INFO(LOG_APP, "IsExposureBiasRange end.");
    float exposureBiasValue = (float)exposureBias;
    ret_ = OH_CaptureSession_GetExposureBiasRange(
        captureSession_, &minExposureBias_, &maxExposureBias_, &step_);
    if (ret_ != CAMERA_OK) {
        OH_LOG_ERROR(LOG_APP, "GetExposureBiasRange failed.");
        return CAMERA_INVALID_ARGUMENT;
    }
    ret_ =
        OH_CaptureSession_SetExposureBias(captureSession_, exposureBiasValue);
    OH_LOG_ERROR(LOG_APP, "OH_CaptureSession_SetExposureBias end.");
    if (ret_ != CAMERA_OK) {
        OH_LOG_ERROR(LOG_APP, "SetExposureBias failed.");
        return CAMERA_INVALID_ARGUMENT;
    }
    ret_ =
        OH_CaptureSession_GetExposureBias(captureSession_, &exposureBiasValue);
    if (ret_ != CAMERA_OK) {
        OH_LOG_ERROR(LOG_APP, "GetExposureBias failed.");
        return CAMERA_INVALID_ARGUMENT;
    }
    OH_LOG_INFO(LOG_APP, "IsExposureBiasRange end.");
    return ret_;
}

// focus mode
Camera_ErrorCode NDKCamera::IsFocusModeSupported(uint32_t mode) {
    Camera_FocusMode focusMode = static_cast<Camera_FocusMode>(mode);
    ret_ = OH_CaptureSession_IsFocusModeSupported(captureSession_, focusMode,
                                                  &isFocusModeSupported_);
    if (ret_ != CAMERA_OK) {
        OH_LOG_ERROR(LOG_APP, "IsFocusModeSupported failed.");
        return CAMERA_INVALID_ARGUMENT;
    }
    return ret_;
}

Camera_ErrorCode NDKCamera::IsFocusMode(uint32_t mode) {
    OH_LOG_INFO(LOG_APP, "IsFocusMode start.");
    Camera_FocusMode focusMode = static_cast<Camera_FocusMode>(mode);
    ret_ = OH_CaptureSession_IsFocusModeSupported(captureSession_, focusMode,
                                                  &isFocusModeSupported_);
    if (ret_ != CAMERA_OK) {
        OH_LOG_ERROR(LOG_APP, "IsFocusModeSupported failed.");
        return CAMERA_INVALID_ARGUMENT;
    }
    ret_ = OH_CaptureSession_SetFocusMode(captureSession_, focusMode);
    if (ret_ != CAMERA_OK) {
        OH_LOG_ERROR(LOG_APP, "SetFocusMode failed.");
        return CAMERA_INVALID_ARGUMENT;
    }
    ret_ = OH_CaptureSession_GetFocusMode(captureSession_, &focusMode);
    if (ret_ != CAMERA_OK) {
        OH_LOG_ERROR(LOG_APP, "GetFocusMode failed.");
        return CAMERA_INVALID_ARGUMENT;
    }
    OH_LOG_INFO(LOG_APP, "IsFocusMode end.");
    return ret_;
}

Camera_ErrorCode NDKCamera::IsFocusPoint(float x, float y) {
    OH_LOG_INFO(LOG_APP, "IsFocusPoint start.");
    Camera_Point focusPoint;
    focusPoint.x = x;
    focusPoint.y = y;
    ret_ = OH_CaptureSession_SetFocusPoint(captureSession_, focusPoint);
    if (ret_ != CAMERA_OK) {
        OH_LOG_ERROR(LOG_APP, "SetFocusPoint failed.");
        return CAMERA_INVALID_ARGUMENT;
    }
    ret_ = OH_CaptureSession_GetFocusPoint(captureSession_, &focusPoint);
    if (ret_ != CAMERA_OK) {
        OH_LOG_ERROR(LOG_APP, "GetFocusPoint failed.");
        return CAMERA_INVALID_ARGUMENT;
    }
    OH_LOG_INFO(LOG_APP, "IsFocusPoint end.");
    return ret_;
}

int32_t NDKCamera::GetVideoFrameWidth(void) {
    videoProfile_ = cameraOutputCapability_->videoProfiles[0];
    if (videoProfile_ == nullptr) {
        OH_LOG_ERROR(LOG_APP, "Get videoProfiles failed.");
        return CAMERA_INVALID_ARGUMENT;
    }
    return videoProfile_->size.width;
}

int32_t NDKCamera::GetVideoFrameHeight(void) {
    videoProfile_ = cameraOutputCapability_->videoProfiles[0];
    if (videoProfile_ == nullptr) {
        OH_LOG_ERROR(LOG_APP, "Get videoProfiles failed.");
        return CAMERA_INVALID_ARGUMENT;
    }
    return videoProfile_->size.height;
}

int32_t NDKCamera::GetVideoFrameRate(void) {
    videoProfile_ = cameraOutputCapability_->videoProfiles[0];
    if (videoProfile_ == nullptr) {
        OH_LOG_ERROR(LOG_APP, "Get videoProfiles failed.");
        return CAMERA_INVALID_ARGUMENT;
    }
    return videoProfile_->range.min;
}

Camera_ErrorCode NDKCamera::VideoOutputStop(void) {
    OH_LOG_ERROR(LOG_APP, "enter VideoOutputStop.");
    ret_ = OH_VideoOutput_Stop(videoOutput_);
    if (ret_ != CAMERA_OK) {
        OH_LOG_ERROR(LOG_APP, "VideoOutputStop failed.");
        return CAMERA_INVALID_ARGUMENT;
    }
    return ret_;
}

Camera_ErrorCode NDKCamera::VideoOutputRelease(void) {
    OH_LOG_ERROR(LOG_APP, "enter VideoOutputRelease.");
    ret_ = OH_VideoOutput_Release(videoOutput_);
    if (ret_ != CAMERA_OK) {
        OH_LOG_ERROR(LOG_APP, "VideoOutputRelease failed.");
        return CAMERA_INVALID_ARGUMENT;
    }
    return ret_;
}

Camera_ErrorCode NDKCamera::TakePicture(void) {
    Camera_ErrorCode ret = CAMERA_OK;
    ret = OH_PhotoOutput_Capture(photoOutput_);
    OH_LOG_ERROR(LOG_APP,
                 "takePicture OH_PhotoOutput_Capture ret = %{public}d.", ret);
    if (ret != CAMERA_OK) {
        OH_LOG_ERROR(LOG_APP, "startPhoto failed.");
        return CAMERA_INVALID_ARGUMENT;
    }
    return ret;
}

Camera_ErrorCode NDKCamera::TakePictureWithPhotoSettings(
    Camera_PhotoCaptureSetting photoSetting) {
    Camera_ErrorCode ret = CAMERA_OK;
    ret = OH_PhotoOutput_Capture_WithCaptureSetting(photoOutput_, photoSetting);

    OH_LOG_INFO(LOG_APP,
                "TakePictureWithPhotoSettings get quality %{public}d, rotation "
                "%{public}d, mirror %{public}d, "
                "latitude, %f, longitude %f, altitude %f",
                photoSetting.quality, photoSetting.rotation,
                photoSetting.mirror, photoSetting.location->latitude,
                photoSetting.location->longitude,
                photoSetting.location->altitude);

    OH_LOG_ERROR(LOG_APP,
                 "takePicture TakePictureWithPhotoSettings ret = %{public}d.",
                 ret);
    if (ret != CAMERA_OK) {
        OH_LOG_ERROR(LOG_APP, "startPhoto failed.");
        return CAMERA_INVALID_ARGUMENT;
    }
    return ret;
}

// CameraManager Callback
void CameraManagerStatusCallback(Camera_Manager *cameraManager,
                                 Camera_StatusInfo *status) {
    OH_LOG_INFO(LOG_APP, "CameraManagerStatusCallback");
}

CameraManager_Callbacks *NDKCamera::GetCameraManagerListener(void) {
    static CameraManager_Callbacks cameraManagerListener = {
        .onCameraStatus = CameraManagerStatusCallback};
    return &cameraManagerListener;
}

Camera_ErrorCode NDKCamera::CameraManagerRegisterCallback(void) {
    ret_ = OH_CameraManager_RegisterCallback(cameraManager_,
                                             GetCameraManagerListener());
    if (ret_ != CAMERA_OK) {
        OH_LOG_ERROR(LOG_APP, "OH_CameraManager_RegisterCallback failed.");
    }
    return ret_;
}

// CameraInput Callback
void OnCameraInputError(const Camera_Input *cameraInput,
                        Camera_ErrorCode errorCode) {
    OH_LOG_INFO(LOG_APP, "OnCameraInput errorCode = %{public}d", errorCode);
}

CameraInput_Callbacks *NDKCamera::GetCameraInputListener(void) {
    static CameraInput_Callbacks cameraInputCallbacks = {
        .onError = OnCameraInputError};
    return &cameraInputCallbacks;
}

Camera_ErrorCode NDKCamera::CameraInputRegisterCallback(void) {
    ret_ =
        OH_CameraInput_RegisterCallback(cameraInput_, GetCameraInputListener());
    if (ret_ != CAMERA_OK) {
        OH_LOG_ERROR(LOG_APP, "OH_CameraInput_RegisterCallback failed.");
    }
    return ret_;
}

// PreviewOutput Callback
void PreviewOutputOnFrameStart(Camera_PreviewOutput *previewOutput) {
    OH_LOG_INFO(LOG_APP, "PreviewOutputOnFrameStart");
}

void PreviewOutputOnFrameEnd(Camera_PreviewOutput *previewOutput,
                             int32_t frameCount) {
    OH_LOG_INFO(LOG_APP, "PreviewOutput frameCount = %{public}d", frameCount);
}

void PreviewOutputOnError(Camera_PreviewOutput *previewOutput,
                          Camera_ErrorCode errorCode) {
    OH_LOG_INFO(LOG_APP, "PreviewOutput errorCode = %{public}d", errorCode);
}

PreviewOutput_Callbacks *NDKCamera::GetPreviewOutputListener(void) {
    static PreviewOutput_Callbacks previewOutputListener = {
        .onFrameStart = PreviewOutputOnFrameStart,
        .onFrameEnd = PreviewOutputOnFrameEnd,
        .onError = PreviewOutputOnError};
    return &previewOutputListener;
}

Camera_ErrorCode NDKCamera::PreviewOutputRegisterCallback(void) {
    ret_ = OH_PreviewOutput_RegisterCallback(previewOutput_,
                                             GetPreviewOutputListener());
    if (ret_ != CAMERA_OK) {
        OH_LOG_ERROR(LOG_APP, "OH_PreviewOutput_RegisterCallback failed.");
    }
    return ret_;
}

// PhotoOutput Callback
void PhotoOutputOnFrameStart(Camera_PhotoOutput *photoOutput) {
    OH_LOG_INFO(LOG_APP, "PhotoOutputOnFrameStart");
}

void PhotoOutputOnFrameShutter(Camera_PhotoOutput *photoOutput,
                               Camera_FrameShutterInfo *info) {
    OH_LOG_INFO(LOG_APP, "PhotoOutputOnFrameShutter");
}

void PhotoOutputOnFrameEnd(Camera_PhotoOutput *photoOutput,
                           int32_t frameCount) {
    OH_LOG_INFO(LOG_APP, "PhotoOutput frameCount = %{public}d", frameCount);
}

void PhotoOutputOnError(Camera_PhotoOutput *photoOutput,
                        Camera_ErrorCode errorCode) {
    OH_LOG_INFO(LOG_APP, "PhotoOutput errorCode = %{public}d", errorCode);
}

PhotoOutput_Callbacks *NDKCamera::GetPhotoOutputListener(void) {
    static PhotoOutput_Callbacks photoOutputListener = {
        .onFrameStart = PhotoOutputOnFrameStart,
        .onFrameShutter = PhotoOutputOnFrameShutter,
        .onFrameEnd = PhotoOutputOnFrameEnd,
        .onError = PhotoOutputOnError};
    return &photoOutputListener;
}

Camera_ErrorCode NDKCamera::PhotoOutputRegisterCallback(void) {
    ret_ =
        OH_PhotoOutput_RegisterCallback(photoOutput_, GetPhotoOutputListener());
    if (ret_ != CAMERA_OK) {
        OH_LOG_ERROR(LOG_APP, "OH_PhotoOutput_RegisterCallback failed.");
    }
    return ret_;
}

// VideoOutput Callback
void VideoOutputOnFrameStart(Camera_VideoOutput *videoOutput) {
    OH_LOG_INFO(LOG_APP, "VideoOutputOnFrameStart");
}

void VideoOutputOnFrameEnd(Camera_VideoOutput *videoOutput,
                           int32_t frameCount) {
    OH_LOG_INFO(LOG_APP, "VideoOutput frameCount = %{public}d", frameCount);
}

void VideoOutputOnError(Camera_VideoOutput *videoOutput,
                        Camera_ErrorCode errorCode) {
    OH_LOG_INFO(LOG_APP, "VideoOutput errorCode = %{public}d", errorCode);
}

VideoOutput_Callbacks *NDKCamera::GetVideoOutputListener(void) {
    static VideoOutput_Callbacks videoOutputListener = {
        .onFrameStart = VideoOutputOnFrameStart,
        .onFrameEnd = VideoOutputOnFrameEnd,
        .onError = VideoOutputOnError};
    return &videoOutputListener;
}

Camera_ErrorCode NDKCamera::VideoOutputRegisterCallback(void) {
    ret_ =
        OH_VideoOutput_RegisterCallback(videoOutput_, GetVideoOutputListener());
    if (ret_ != CAMERA_OK) {
        OH_LOG_ERROR(LOG_APP, "OH_VideoOutput_RegisterCallback failed.");
    }
    return ret_;
}

// Metadata Callback
void OnMetadataObjectAvailable(Camera_MetadataOutput *metadataOutput,
                               Camera_MetadataObject *metadataObject,
                               uint32_t size) {
    OH_LOG_INFO(LOG_APP, "size = %{public}d", size);
}

void OnMetadataOutputError(Camera_MetadataOutput *metadataOutput,
                           Camera_ErrorCode errorCode) {
    OH_LOG_INFO(LOG_APP, "OnMetadataOutput errorCode = %{public}d", errorCode);
}

MetadataOutput_Callbacks *NDKCamera::GetMetadataOutputListener(void) {
    static MetadataOutput_Callbacks metadataOutputListener = {
        .onMetadataObjectAvailable = OnMetadataObjectAvailable,
        .onError = OnMetadataOutputError};
    return &metadataOutputListener;
}

Camera_ErrorCode NDKCamera::MetadataOutputRegisterCallback(void) {
    ret_ = OH_MetadataOutput_RegisterCallback(metadataOutput_,
                                              GetMetadataOutputListener());
    if (ret_ != CAMERA_OK) {
        OH_LOG_ERROR(LOG_APP, "OH_MetadataOutput_RegisterCallback failed.");
    }
    return ret_;
}

// Session Callback
void CaptureSessionOnFocusStateChange(Camera_CaptureSession *session,
                                      Camera_FocusState focusState) {
    OH_LOG_INFO(LOG_APP, "CaptureSessionOnFocusStateChange");
}

void CaptureSessionOnError(Camera_CaptureSession *session,
                           Camera_ErrorCode errorCode) {
    OH_LOG_INFO(LOG_APP, "CaptureSession errorCode = %{public}d", errorCode);
}

CaptureSession_Callbacks *NDKCamera::GetCaptureSessionRegister(void) {
    static CaptureSession_Callbacks captureSessionCallbacks = {
        .onFocusStateChange = CaptureSessionOnFocusStateChange,
        .onError = CaptureSessionOnError};
    return &captureSessionCallbacks;
}

Camera_ErrorCode NDKCamera::CaptureSessionRegisterCallback(void) {
    ret_ = OH_CaptureSession_RegisterCallback(captureSession_,
                                              GetCaptureSessionRegister());
    if (ret_ != CAMERA_OK) {
        OH_LOG_ERROR(LOG_APP, "OH_CaptureSession_RegisterCallback failed.");
    }
    return ret_;
}
} // namespace bmf_lite_demo