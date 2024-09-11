/*
 * Copyright 2023 Babit Authors
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
#ifndef _BMF_ALGORITHM_MODULES_OPS_OPENGL_DENOISE_H_
#define _BMF_ALGORITHM_MODULES_OPS_OPENGL_DENOISE_H_
#include <string>
#include "opengl/gl_helper.h"

namespace bmf_lite {
namespace opengl {

/**
 * @brief Denoise using spatial bilateral and temporal mix filter
 *
 */
class Denoise {
  public:
    int init(const std::string &program_cache_dir);
    /**
     * @brief
     *
     * @param in_tex texture2D with RGBA8 format and content
     * @param out_tex texture2D with RGBA8 format and content
     * @param width
     * @param height
     * @return true
     * @return false
     */
    int run(GLuint in_tex, GLuint out_tex, int width, int height);

    ~Denoise();

  private:
    GLuint program_ = GL_NONE;
    GLuint program_no_mix_ = GL_NONE;

    int in_width_ = 0;
    int in_height_ = 0;

    GLint local_size_x_ = 0;
    GLint local_size_y_ = 0;

    GLuint tex_ping_ = GL_NONE;
    GLuint tex_pong_ = GL_NONE;

    bool first_run_ = true;
    bool inited_ = false;
}; // Denoise

} // namespace opengl
} // namespace bmf_lite
#endif
