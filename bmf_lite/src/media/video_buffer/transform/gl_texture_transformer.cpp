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

#ifdef BMF_LITE_ENABLE_CPUMEMORYBUFFER
#ifdef BMF_LITE_ENABLE_OPENGLTEXTUREBUFFER
#include "gl_texture_transformer.h"
#include "common/error_code.h"
#include <EGL/egl.h>
#include <GLES3/gl3.h>
#include <iostream>
#include "opengl/resize/resize.h"
namespace bmf_lite {

class GLTextureTransformerImpl {
  public:
    std::shared_ptr<HWDeviceContext> device_context_;
    bool inited_ = false;
    unsigned int fbo_ = 0;
    unsigned char *byte_data_ptr_ = nullptr;
    float *float_data_ptr_ = nullptr;
    unsigned char *cache_data_ptr_ = nullptr;
    GLuint resize_texture_id = 0;
    int need_resize = 0;
    std::shared_ptr<opengl::Resize> resize_ = nullptr;
    ~GLTextureTransformerImpl() {}
};

GLTextureTransformer::GLTextureTransformer() {}

int GLTextureTransformer::init(
    HardwareDataInfo hardware_data_info_in,
    std::shared_ptr<HWDeviceContext> device_context) {
    impl_ = std::make_shared<GLTextureTransformerImpl>();
    impl_->inited_ = true;
    return BMF_LITE_StsOk;
}

int GLTextureTransformer::transTexture2Memory(
    std::shared_ptr<VideoBuffer> in_video_buffer,
    std::shared_ptr<VideoBuffer> &out_video_buffer) {
    if (impl_->inited_ &&
        in_video_buffer->memoryType() == MemoryType::kOpenGLTexture2d &&
        out_video_buffer->memoryType() == MemoryType::kByteMemory) {
        int width = in_video_buffer->width();
        int height = in_video_buffer->height();
        int in_tex = (long)(in_video_buffer->data());
        int out_width = out_video_buffer->width();
        int out_height = out_video_buffer->height();
        impl_->need_resize = 0;
        if (impl_->byte_data_ptr_ == nullptr) {
            impl_->byte_data_ptr_ =
                (new unsigned char[out_width * out_height *
                                   4]); // Large model with fixed width and
                                        // height
        }
        if (out_width != width || out_height != height) {
            impl_->need_resize = 1;
            if (impl_->resize_texture_id == 0) {
                glGenTextures(1, &impl_->resize_texture_id);
                glBindTexture(GL_TEXTURE_2D, impl_->resize_texture_id);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                                GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
                                GL_LINEAR);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, out_width, out_height,
                             0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
            }
            if (impl_->resize_ == nullptr) {
                impl_->resize_ = std::make_shared<opengl::Resize>();
                impl_->resize_->init("");
            }
        }
        if (impl_->need_resize == 1) {
            impl_->resize_->run(in_tex, impl_->resize_texture_id, width, height,
                                out_width, out_height);
            in_tex = impl_->resize_texture_id;
        }
        glBindTexture(GL_TEXTURE_2D, in_tex);
        if (impl_->fbo_ == 0) {
            glGenFramebuffers(1, &impl_->fbo_);
        }
        glBindFramebuffer(GL_FRAMEBUFFER, impl_->fbo_);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_2D, in_tex, 0);
        glReadPixels(0, 0, out_width, out_height, GL_RGBA, GL_UNSIGNED_BYTE,
                     impl_->byte_data_ptr_);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        if (out_video_buffer->memoryType() == MemoryType::kByteMemory &&
            out_video_buffer->hardwareDataInfo().internal_format ==
                bmf_lite::CPU_RGBFLOAT) {
            float *tmp_ptr = (float *)out_video_buffer->data();
            for (int i = 0; i < out_height; i++) {
                for (int j = 0; j < out_width; j++) { // 4 channel to 3 channel
                    tmp_ptr[i * out_width * 3 + j * 3] =
                        float(
                            impl_->byte_data_ptr_[i * out_width * 4 + j * 4]) /
                        255.0f;
                    tmp_ptr[i * out_width * 3 + j * 3 + 1] =
                        float(impl_->byte_data_ptr_[i * out_width * 4 + j * 4 +
                                                    1]) /
                        255.0f;
                    tmp_ptr[i * out_width * 3 + j * 3 + 2] =
                        float(impl_->byte_data_ptr_[i * out_width * 4 + j * 4 +
                                                    2]) /
                        255.0f;
                }
            }
        } else {
            return BMF_LITE_StsBadArg;
        }
    }
    return BMF_LITE_StsOk;
}

int GLTextureTransformer::transMemory2Texture(
    std::shared_ptr<VideoBuffer> in_video_buffer,
    std::shared_ptr<VideoBuffer> out_video_buffer) {
    if (impl_->inited_ &&
        out_video_buffer->memoryType() == MemoryType::kOpenGLTexture2d &&
        in_video_buffer->memoryType() == MemoryType::kByteMemory) {
        int width = in_video_buffer->width();
        int height = in_video_buffer->height();
        int out_width = out_video_buffer->width();
        int out_height = out_video_buffer->height();
        int texutre = (long)(out_video_buffer->data());
        if (impl_->cache_data_ptr_ == nullptr) {
            impl_->cache_data_ptr_ =
                (new unsigned char[width * height *
                                   4]); // Large model with fixed width and
                                        // height
        }
        impl_->need_resize = 0;
        if (in_video_buffer->memoryType() == MemoryType::kByteMemory &&
            in_video_buffer->hardwareDataInfo().internal_format ==
                bmf_lite::CPU_RGBFLOAT) {
            float *memory_data = (float *)in_video_buffer->data();
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) { // 4 channel to 3 channel
                    impl_->cache_data_ptr_[i * width * 4 + j * 4] =
                        (unsigned char)(255.f *
                                        memory_data[i * width * 3 + j * 3]);
                    impl_->cache_data_ptr_[i * width * 4 + j * 4 + 1] =
                        (unsigned char)(255.f *
                                        memory_data[i * width * 3 + j * 3 + 1]);
                    impl_->cache_data_ptr_[i * width * 4 + j * 4 + 2] =
                        (unsigned char)(255.f *
                                        memory_data[i * width * 3 + j * 3 + 2]);
                    impl_->cache_data_ptr_[i * width * 4 + j * 4 + 3] = 255;
                }
            }
        } else {
            return BMF_LITE_StsBadArg;
        }

        if (texutre > 0) {
            int tex_id = texutre;
            if (out_width != width || out_height != height) {
                impl_->need_resize = 1;
                if (impl_->resize_texture_id == 0) {
                    glGenTextures(1, &impl_->resize_texture_id);
                    glBindTexture(GL_TEXTURE_2D, impl_->resize_texture_id);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,
                                    GL_REPEAT);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,
                                    GL_REPEAT);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                                    GL_LINEAR);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
                                    GL_LINEAR);
                    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, out_width,
                                 out_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
                }
                tex_id = impl_->resize_texture_id;
                if (impl_->resize_ == nullptr) {
                    impl_->resize_ = std::make_shared<opengl::Resize>();
                    impl_->resize_->init("");
                }
            }
            glBindTexture(GL_TEXTURE_2D, tex_id);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA,
                            GL_UNSIGNED_BYTE, impl_->cache_data_ptr_);
            if (impl_->need_resize == 1) {
                impl_->resize_->run(tex_id, texutre, width, height, out_width,
                                    out_height);
            }
        } else {
            return BMF_LITE_StsBadArg;
        }
    } else {
        return BMF_LITE_StsBadArg;
    }
    return BMF_LITE_StsOk;
}

GLTextureTransformer::~GLTextureTransformer() {
    if (impl_->fbo_ != 0) {
        glDeleteFramebuffers(1, &impl_->fbo_);
        impl_->fbo_ = 0;
    }
    if (impl_->cache_data_ptr_ != nullptr) {
        delete[] impl_->cache_data_ptr_;
        impl_->cache_data_ptr_ = nullptr;
    }
    if (impl_->byte_data_ptr_ != nullptr) {
        delete[] impl_->byte_data_ptr_;
        impl_->byte_data_ptr_ = nullptr;
    }
    if (impl_->resize_texture_id != 0) {
        glDeleteTextures(1, (unsigned int *)&(impl_->resize_texture_id));
    }
}
} // namespace bmf_lite
#endif
#endif
