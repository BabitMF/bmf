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
#ifndef CAMERA_NDK_CAMERA_H
#define CAMERA_NDK_CAMERA_H

#include <unistd.h>
#include <string>
#include <thread>
#include <cstdio>
#include <fcntl.h>
#include <map>
#include <string>
#include <vector>
#include <native_buffer/native_buffer.h>
#include "iostream"
#include <mutex>

#include "hilog/log.h"
#include "ohcamera/camera.h"
#include "ohcamera/camera_input.h"
#include "ohcamera/capture_session.h"
#include "ohcamera/photo_output.h"
#include "ohcamera/preview_output.h"
#include "ohcamera/video_output.h"
#include "napi/native_api.h"
#include "ohcamera/camera_manager.h"

namespace bmf_lite_demo {
class NDKCamera {
  public:
    ~NDKCamera();
    NDKCamera(const char *surfaceId, uint32_t focusMode,
              uint32_t cameraDeviceIndex);

    //     static void Destroy()
    //     {
    //         if (NDKCamera::ndkCamera_ != nullptr) {
    //             delete NDKCamera::ndkCamera_;
    //             NDKCamera::ndkCamera_ = nullptr;
    //         }
    //     }

    Camera_ErrorCode CreateCameraInput(void);
    Camera_ErrorCode CameraInputOpen(void);
    Camera_ErrorCode CameraInputClose(void);
    Camera_ErrorCode CameraInputRelease(void);
    Camera_ErrorCode GetSupportedCameras(void);
    Camera_ErrorCode GetSupportedOutputCapability(void);
    Camera_ErrorCode CreatePreviewOutput(void);
    Camera_ErrorCode CreatePhotoOutput(char *photoId);
    Camera_ErrorCode CreateVideoOutput(const char *videoId);
    Camera_ErrorCode CreateMetadataOutput(void);
    Camera_ErrorCode IsCameraMuted(void);
    Camera_ErrorCode PreviewOutputStop(void);
    Camera_ErrorCode PreviewOutputRelease(void);
    Camera_ErrorCode PhotoOutputRelease(void);
    Camera_ErrorCode HasFlashFn(uint32_t mode);
    Camera_ErrorCode IsVideoStabilizationModeSupportedFn(uint32_t mode);
    Camera_ErrorCode setZoomRatioFn(uint32_t zoomRatio);
    Camera_ErrorCode SessionFlowFn(void);
    Camera_ErrorCode SessionBegin(void);
    Camera_ErrorCode SessionCommitConfig(void);
    Camera_ErrorCode SessionStart(void);
    Camera_ErrorCode SessionStop(void);
    Camera_ErrorCode StartVideo(char *videoId, char *photoId);
    Camera_ErrorCode AddVideoOutput(void);
    Camera_ErrorCode AddPhotoOutput();
    Camera_ErrorCode VideoOutputStart(void);
    Camera_ErrorCode StartPhoto(char *mSurfaceId);
    Camera_ErrorCode IsExposureModeSupportedFn(uint32_t mode);
    Camera_ErrorCode IsMeteringPoint(int x, int y);
    Camera_ErrorCode IsExposureBiasRange(int exposureBias);
    Camera_ErrorCode IsFocusMode(uint32_t mode);
    Camera_ErrorCode IsFocusPoint(float x, float y);
    Camera_ErrorCode IsFocusModeSupported(uint32_t mode);
    Camera_ErrorCode ReleaseCamera(void);
    Camera_ErrorCode SessionRelease(void);
    Camera_ErrorCode ReleaseSession(void);
    int32_t GetVideoFrameWidth(void);
    int32_t GetVideoFrameHeight(void);
    int32_t GetVideoFrameRate(void);
    Camera_ErrorCode VideoOutputStop(void);
    Camera_ErrorCode VideoOutputRelease(void);
    Camera_ErrorCode TakePicture(void);
    Camera_ErrorCode
    TakePictureWithPhotoSettings(Camera_PhotoCaptureSetting photoSetting);
    // callback
    Camera_ErrorCode CameraManagerRegisterCallback(void);
    Camera_ErrorCode CameraInputRegisterCallback(void);
    Camera_ErrorCode PreviewOutputRegisterCallback(void);
    Camera_ErrorCode PhotoOutputRegisterCallback(void);
    Camera_ErrorCode VideoOutputRegisterCallback(void);
    Camera_ErrorCode MetadataOutputRegisterCallback(void);
    Camera_ErrorCode CaptureSessionRegisterCallback(void);

    // Get callback
    CameraManager_Callbacks *GetCameraManagerListener(void);
    CameraInput_Callbacks *GetCameraInputListener(void);
    PreviewOutput_Callbacks *GetPreviewOutputListener(void);
    PhotoOutput_Callbacks *GetPhotoOutputListener(void);
    VideoOutput_Callbacks *GetVideoOutputListener(void);
    MetadataOutput_Callbacks *GetMetadataOutputListener(void);
    CaptureSession_Callbacks *GetCaptureSessionRegister(void);

  private:
    NDKCamera(const NDKCamera &) = delete;
    NDKCamera &operator=(const NDKCamera &) = delete;
    uint32_t cameraDeviceIndex_;
    Camera_Manager *cameraManager_;
    Camera_CaptureSession *captureSession_;
    Camera_Device *cameras_;
    uint32_t size_;
    Camera_OutputCapability *cameraOutputCapability_;
    const Camera_Profile *profile_;
    const Camera_VideoProfile *videoProfile_;
    Camera_PreviewOutput *previewOutput_ = nullptr;
    Camera_PhotoOutput *photoOutput_ = nullptr;
    Camera_VideoOutput *videoOutput_ = nullptr;
    const Camera_MetadataObjectType *metaDataObjectType_;
    Camera_MetadataOutput *metadataOutput_;
    Camera_Input *cameraInput_;
    bool *isCameraMuted_;
    Camera_Position position_;
    Camera_Type type_;
    const char *previewSurfaceId_;
    char *photoSurfaceId_;
    Camera_ErrorCode ret_;
    uint32_t takePictureTimes = 0;
    Camera_ExposureMode exposureMode_;
    bool isExposureModeSupported_;
    bool isFocusModeSupported_;
    float minExposureBias_;
    float maxExposureBias_;
    float step_;
    uint32_t focusMode_;

    //     static NDKCamera *ndkCamera_;
    static std::mutex mtx_;
    volatile bool valid_;
};
} // namespace bmf_lite_demo
#endif // CAMERA_NDK_CAMERA_H
