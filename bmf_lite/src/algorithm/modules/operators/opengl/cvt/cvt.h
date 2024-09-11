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
#ifndef _BMF_ALGORITHM_MODULES_OPS_OPENGL_CVT_H_
#define _BMF_ALGORITHM_MODULES_OPS_OPENGL_CVT_H_

#include <string>
#include "opengl/gl_helper.h"

namespace bmf_lite {
namespace opengl {
class Cvt {
  public:
    enum class ColorSpace {
        UNKNOWN = -1,
        BT601 = 0,
        BT709 = 1,
    };

    enum class ColorRange {
        UNKNOWN = -1,
        LIMITED = 0,
        FULL = 1,
    };

    enum class CvtMode {
        UNKNOWN = -1,
        RGBA_TO_YUV420P = 0,
        YUV420P_TO_RGBA = 1,
        RGBA_TO_YUV444P = 2,
        YUV444P_TO_RGBA = 3,
        RGBA_TO_YUVA444 = 4,
        YUVA444_TO_RGBA = 5,
        RGBA_TO_HSVA = 6,
        HSVA_TO_RGBA = 7,
    };

  public:
    int init(CvtMode mode, const std::string &program_cache_dir);
    /**
     * @brief rgba convert to yuv444 + a, rgba convert to hsva, hsva convert to
     * rgba
     *
     * @param in_tex
     * @param out_tex
     * @param width
     * @param height
     * @param cs N/A for hsva
     * @param in_cr N/A for hsva
     * @param out_cr N/A for hsva
     * @return true
     * @return false
     */
    int run(GLuint in_tex, GLuint out_tex, int width, int height,
            ColorSpace cs = ColorSpace::BT601,
            ColorRange in_cr = ColorRange::LIMITED,
            ColorRange out_cr = ColorRange::LIMITED);

    /**
     * @brief convert yuv420p to rgba or vice versa, yuv444p to rgba or vice
     * versa vaild for CvtMode： RGBA_TO_YUV420P, YUV420P_TO_RGBA,
     * RGBA_TO_YUV444P, YUV444P_TO_RGBA tex0, tex0, tex1, tex3 order follow the
     * rules:
     * 1. output texture comes after input texture
     * 2. for y/u/v, y first, u second, v last
     * @param tex0
     * @param tex1
     * @param tex2
     * @param tex3
     * @param width
     * @param height
     * @param cs
     * @param in_cr
     * @param out_cr
     * @return true
     * @return false
     */
    int run(GLuint tex0, GLuint tex1, GLuint tex2, GLuint tex3, int width,
            int height, ColorSpace cs, ColorRange in_cr, ColorRange out_cr);
    ~Cvt();

  private:
    /**
     * @brief rgba convert to yuv420
     *
     * @param in_tex
     * @param out_y
     * @param out_u
     * @param out_v
     * @param width
     * @param height
     * @param cs
     * @param in_cr
     * @param out_cr
     * @return true
     * @return false
     */
    int run_1to3(GLuint in_tex, GLuint out_y, GLuint out_u, GLuint out_v,
                 int width, int height, ColorSpace cs, ColorRange in_cr,
                 ColorRange out_cr, int in_uv_width = 0, int in_uv_height = 0);
    /**
     * @brief yuv420 convert to rgba, yuv444 convert to rgba
     *
     * @param in_y
     * @param in_u
     * @param in_v
     * @param out_tex
     * @param width
     * @param height
     * @param cs
     * @param in_cr
     * @param out_cr
     * @return true
     * @return false
     */
    int run_3to1(GLuint in_y, GLuint in_u, GLuint in_v, GLuint out_tex,
                 int width, int height, ColorSpace cs, ColorRange in_cr,
                 ColorRange out_cr, int in_uv_width = 0, int in_uv_height = 0);

  private:
    CvtMode cvt_mode_ = CvtMode::UNKNOWN;
    ColorSpace cs_ = ColorSpace::UNKNOWN;
    ColorRange in_cr_ = ColorRange::UNKNOWN;
    ColorRange out_cr_ = ColorRange::UNKNOWN;
    const float *cm_ = nullptr;
    const float *co_ = nullptr;

    int width_ = 0;
    int height_ = 0;

    GLuint program_id_ = GL_NONE;

    GLint cm_loc_ = 0;
    GLint co_loc_ = 0;
    GLint local_size_[3] = {16, 16, 1};

    GLint num_groups_x_ = 0;
    GLint num_groups_y_ = 0;
};

} // namespace opengl
} // namespace bmf_lite

#endif
