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
#include "algorithm.h"
#include "algorithm/bmf_algorithm_impl.h"

namespace bmf_lite_demo {

Algorithm::Algorithm(EGLDisplay eglDisplay, EGLSurface surface,
                     EGLContext eglContext, int algorithmType)
    : eglDisplay_(eglDisplay), eglSurface_(surface), eglContext_(eglContext),
      algorithmType_(algorithmType) {
    algorithm_ = bmf_lite::AlgorithmFactory::createAlgorithmInterface();
    bmf_lite::Param init_param;
    // init param
    init_param.setInt("change_mode",
                      bmf_lite::ModsMode::CREATE_AND_PROCESS_MODE);
    init_param.setString("instance_id", "i");
    init_param.setInt("algorithm_type", algorithmType_);
    init_param.setInt("algorithm_version", 0);
    init_param.setInt("backend", 3);
    init_param.setInt("scale_mode", 0);
    init_param.setInt("process_mode", 0);
    init_param.setInt("max_width", 1920);
    init_param.setInt("max_height", 1080);
    init_param.setString("license_module_name", "");
    init_param.setString("program_cache_dir", "");
    init_param.setInt("sharp_levels", 0);    // for super_resolution only
    init_param.setString("weight_path", ""); // for super_resolution only

    int init_result = algorithm_->setParam(init_param);
    if (init_result != 0) {
        bmf_lite::AlgorithmFactory::releaseAlgorithmInterface(algorithm_);
        algorithm_ = nullptr;
        valid_ = false;
    }
    valid_ = true;
}

int Algorithm::processVideoFrame(GLuint textureId, size_t width,
                                 size_t height) {
    if (!valid_) {
        return -1;
    }

    bmf_lite::HardwareDeviceSetInfo set_info;
    set_info.device_type = bmf_lite::kHWDeviceTypeEGLCtx;
    bmf_lite::EGLContextInfo egl_context_info_create{eglDisplay_, eglContext_,
                                                     eglSurface_, eglSurface_};
    set_info.context_info = &egl_context_info_create;
    set_info.owned = 0;

    std::shared_ptr<bmf_lite::HWDeviceContext> device_context;
    bmf_lite::HWDeviceContextManager::setHwDeviceContext(&set_info,
                                                         device_context);
    bmf_lite::HardwareDataInfo data_info = {
        bmf_lite::MemoryType::kOpenGLTexture2d, bmf_lite::GLES_TEXTURE_RGBA, 0};

    std::shared_ptr<bmf_lite::VideoBuffer> videoBuffer;
    bmf_lite::VideoBufferManager::createTextureVideoBufferFromExistingData(
        (void *)textureId, width, height, &data_info, device_context, NULL,
        videoBuffer);
    bmf_lite::VideoFrame videoFrame(videoBuffer);

    bmf_lite::Param param;
    param.setInt("sharp_level", 0); // for super_resolution only
    param.setInt("scale_mode", 0);  // for super_resolution only
    return algorithm_->processVideoFrame(videoFrame, param);
}

int Algorithm::getVideoFrameOutput(GLuint &textureId, size_t &width,
                                   size_t &height) {
    if (!valid_) {
        return -1;
    }

    bmf_lite::Param output_param;
    bmf_lite::VideoFrame output_video_frame;
    algorithm_->getVideoFrameOutput(output_video_frame, output_param);
    width = output_video_frame.buffer()->width();
    height = output_video_frame.buffer()->height();
    textureId = (long)(output_video_frame.buffer()->data());
    return 0;
}

Algorithm::~Algorithm() {
    bmf_lite::AlgorithmFactory::releaseAlgorithmInterface(algorithm_);
    valid_ = false;
}

} // namespace bmf_lite_demo