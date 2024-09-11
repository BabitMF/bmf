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
#ifndef _BMF_ALGORITHM_MODULES_OPS_OPENGL_ROTATE_H_
#define _BMF_ALGORITHM_MODULES_OPS_OPENGL_ROTATE_H_
#include <string>
#include "opengl/gl_helper.h"

namespace bmf_lite {
namespace opengl {

/**
 * @brief Rotate
 *
 */
class Rotate {
  public:
    int init(const std::string &program_cache_dir);
    /**
     * @brief
     *
     * @param in_tex
     * @param out_tex
     * @param width
     * @param height
     * @param angle the clock-wise rotate degree, 90, 180, 270 are supported
     * @return error code
     */
    int run(GLuint in_tex, GLuint out_tex, int width, int height, int angle);

    ~Rotate();

  private:
    GLuint program_id_ = GL_NONE;

    GLint local_size_[3] = {16, 16, 1};

    bool inited_ = false;
};

} // namespace opengl
} // namespace bmf_lite
#endif
