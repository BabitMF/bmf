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
#ifndef _BMF_ALGORITHM_MODULES_OPS_OPENGL_RESIZE_H_
#define _BMF_ALGORITHM_MODULES_OPS_OPENGL_RESIZE_H_
#include <string>
#include "opengl/gl_helper.h"

namespace bmf_lite {
namespace opengl {

/**
 * @brief Resize
 *
 */
class Resize {
  public:
    enum class ColorSpace {
        UNKNOW = -1,
        BT601 = 0,
        BT709 = 1,
    };

    enum class ColorRange {
        UNKNOW = -1,
        LIMITED = 0,
        FULL = 1,
    };

  public:
    int init(const std::string &program_cache_dir = "");
    /**
     * @brief rgba convert to yuv444 + a
     *
     * @param in_rgba
     * @param out_rgba
     * @param in_width
     * @param in_height
     * @param out_width
     * @param out_height
     * @return error code
     */
    int run(GLuint in_rgba, GLuint out_rgba, int in_width, int in_height,
            int out_width, int out_height);
    ~Resize();

  private:
    ColorSpace cs_ = ColorSpace::UNKNOW;
    ColorRange in_cr_ = ColorRange::UNKNOW;
    ColorRange out_cr_ = ColorRange::UNKNOW;

    float *cm_ = nullptr;
    float *co_ = nullptr;

    int width_ = 0;
    int height_ = 0;

    GLuint program_id_ = GL_NONE;

    GLint cm_loc_ = 0;
    GLint co_loc_ = 0;
    GLint local_size_[3] = {16, 16, 1};

    GLint num_groups_x_ = 0;
    GLint num_groups_y_ = 0;

    bool inited_ = false;
};

} // namespace opengl
} // namespace bmf_lite
#endif
