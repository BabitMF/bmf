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

#include "gl_texture_video_buffer_allocator.h"
#include "gl_texture_video_buffer.h"
#include <EGL/egl.h>
#include <GLES3/gl3.h>
#include <iostream>

namespace bmf_lite {
#define GL_CHECK_RETURN(FUNC)                                                  \
    FUNC;                                                                      \
    {                                                                          \
        GLenum glError = glGetError();                                         \
        if (glError != GL_NO_ERROR) {                                          \
            std::cout << "Call " << #FUNC << "failed error code:" << glError   \
                      << std::endl;                                            \
            return -1;                                                         \
        }                                                                      \
    }

GlTextureVideoBufferAllocator::GlTextureVideoBufferAllocator() {}

GlTextureVideoBufferAllocator::~GlTextureVideoBufferAllocator() {}

int GlTextureVideoBufferAllocator::allocVideoBuffer(
    int width, int height, HardwareDataInfo *data_info,
    std::shared_ptr<HWDeviceContext> device_context,
    VideoBuffer *&video_buffer) {
    // printf("device_guard\n");
    // HWDeviceContextGuard device_guard(device_context);
    // printf("device_guard end\n");

    GLuint texture_id = 0;
    GL_CHECK_RETURN(glGenTextures(1, &texture_id));
    GL_CHECK_RETURN(glBindTexture(GL_TEXTURE_2D, texture_id));

    // set the texture wrapping parameters
    GL_CHECK_RETURN(
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
    GL_CHECK_RETURN(
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));

    GL_CHECK_RETURN(
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
    GL_CHECK_RETURN(
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
    if (data_info->mutable_flag != 0) {
        int internal_format = 0;
        if (data_info->internal_format == GLES_TEXTURE_RGBA) {
            internal_format = GL_RGBA8;
        }
        if (data_info->internal_format == GLES_TEXTURE_RGBA8UI) {
            internal_format = GL_RGBA8UI;
        }
        // printf("glTexStorage2D\n");
        GL_CHECK_RETURN(
            glTexStorage2D(GL_TEXTURE_2D, 1, internal_format, width, height));
    } else {
        int internal_format = 0;
        int foramt = 0;
        if (data_info->internal_format == GLES_TEXTURE_RGBA) {
            internal_format = GL_RGBA;
            foramt = GL_RGBA;
        }
        // std::cout<<" internal_format:"<<internal_format<<"
        // foramt:"<<foramt<<" width:"<<width<<" height:"<<height<<std::endl;
        GL_CHECK_RETURN(glTexImage2D(GL_TEXTURE_2D, 0, internal_format, width,
                                     height, 0, foramt, GL_UNSIGNED_BYTE, 0));
    }
    (glBindTexture(GL_TEXTURE_2D, 0));
    // std::cout<<"texture id:"<<texture_id<<std::endl;
    video_buffer = new GlTextureVideoBuffer(texture_id, width, height,
                                            *data_info, device_context);
    if (video_buffer != NULL) {
        video_buffer->setDeleter([](VideoBuffer *video_buffer) {
            GlTextureVideoBufferAllocator::releaseVideoBuffer(video_buffer);
        });
        return 0;
    }
    return 0;
}

int GlTextureVideoBufferAllocator::releaseVideoBuffer(
    VideoBuffer *video_buffer) {
    std::shared_ptr<HWDeviceContext> device_context =
        video_buffer->getHWDeviceContext();
    // HWDeviceContextGuard device_guard(device_context);
    if (video_buffer != NULL &&
        video_buffer->memoryType() == MemoryType::kOpenGLTexture2d) {
        GlTextureVideoBuffer *gl_texture_video_buffer =
            (GlTextureVideoBuffer *)video_buffer;
        GLuint texture_id = gl_texture_video_buffer->getTextureId();
        if (texture_id != 0) {
            glDeleteTextures(1, &texture_id);
        }
    }
    video_buffer = NULL;
    return 0;
}

} // namespace bmf_lite
#endif