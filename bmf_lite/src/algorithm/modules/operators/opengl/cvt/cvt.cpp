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
#include "cvt.h"

namespace bmf_lite {
namespace opengl {

constexpr float cm_rgb2yuv_bt601_l2l[9] = {
    0.299, 0.587, 0.114, -0.172, -0.339, 0.511, 0.511 - 0.428, -0.083};
constexpr float co_rgb2yuv_bt601_l2l[3] = {0, 0.5, 0.5};

constexpr float cm_yuv2rgb_bt601_l2l[9] = {1,      0, 1.371, 1, -0.336,
                                           -0.698, 1, 1.732, 0};
constexpr float co_yuv2rgb_bt601_l2l[3] = {0, -0.5, -0.5};

constexpr float cm_rgb2yuv_bt601_f2l[9] = {0.257, 0.504, 0.098,  -0.148, -0.291,
                                           0.439, 0.439, -0.368, -0.071};
constexpr float co_rgb2yuv_bt601_f2l[3] = {0.0625, 0.5, 0.5};

constexpr float cm_yuv2rgb_bt601_l2f[9] = {1.164,  0,     1.596, 1.164, -0.391,
                                           -0.813, 1.164, 2.018, 0};
constexpr float co_yuv2rgb_bt601_l2f[3] = {-0.0625, -0.5, -0.5};

constexpr float cm_yuv2rgb_bt601_f2f[9] = {1,       0, 1.402, 1, -0.3441f,
                                           -0.7141, 1, 1.772, 0};
constexpr float co_yuv2rgb_bt601_f2f[3] = {0, -0.5, -0.5};

constexpr float cm_yuv2rgb_bt709_l2f[9] = {1.164,  0,     1.793, 1.164, -0.213,
                                           -0.534, 1.164, 2.115, 0};
constexpr float co_yuv2rgb_bt709_l2f[3] = {-0.0625, -0.5, -0.5};

constexpr float cm_yuv2rgb_bt709_f2f[9] = {1,       0, 1.5748, 1, -0.1873,
                                           -0.4681, 1, 1.8556, 0};
constexpr float co_yuv2rgb_bt709_f2f[3] = {0, -0.5, -0.5};

constexpr float cm_rgb2yuv_bt709_f2l[9] = {0.183, 0.614, 0.062,  -0.101, -0.338,
                                           0.439, 0.439, -0.399, -0.040};
constexpr float co_rgb2yuv_bt709_f2l[3] = {0.0625, 0.5, 0.5};

constexpr float cm_yuv2rgb_bt709_l2l[9] = {1,      0, 1.540, 1, -0.183,
                                           -0.459, 1, 1.816, 0};
constexpr float co_yuv2rgb_bt709_l2l[3] = {0, -0.5, -0.5};

constexpr float cm_rgb2yuv_bt709_l2l[9] = {0.213, 0.715, 0.072,  -0.117, -0.394,
                                           0.511, 0.511, -0.464, -0.047};
constexpr float co_rgb2yuv_bt709_l2l[3] = {0, 0.5, 0.5};

constexpr float cm_yuv2rgb_bt2020_l2f[9] = {
    1.1644, 0, 1.6787, 1.1644, -0.1873, -0.6504, 1.1644, 2.1418, 0};
constexpr float co_yuv2rgb_bt2020_l2f[3] = {-0.0625, -0.5, -0.5};

constexpr float cm_yuv2rgb_bt2020_f2f[9] = {1,       0, 1.4746, 1, -0.1646,
                                            -0.5714, 1, 1.8814, 0};
constexpr float co_yuv2rgb_bt2020_f2f[3] = {0, -0.5, -0.5};

constexpr float cm_yuv2rgb_jpeg[9] = {
    1, -0.00093, 1.401687, 1, -0.3437, -0.71417, 1, 1.77216, 0.00099};
constexpr float co_yuv2rgb_jpeg[3] = {0, -0.5, -0.5};

constexpr float cm_rgb2yuv_jpeg[9] = {0.299,  0.587, 0.114, -0.169,
                                      -0.331, 0.5,   0.5,   -0.419 - 0.081};
constexpr float co_rgb2yuv_jpeg[3] = {0, 0.5, 0.5};

static constexpr char rgba_to_yuv444p[] = R"(
layout (location = 0) uniform mediump sampler2D in_img; //rgba8
layout (rgba8, binding = 1) writeonly uniform mediump image2D out_y;
layout (rgba8, binding = 2) writeonly uniform mediump image2D out_u;
layout (rgba8, binding = 3) writeonly uniform mediump image2D out_v;
layout (location=4) uniform ivec2 in_img_size;

uniform mat3 cm;
uniform vec3 co;

void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    int width = in_img_size.x;
    int height = in_img_size.y;

    if (pos.x < width && pos.y < height) {
        vec4 rgba = texelFetch(in_img, pos, 0);
        vec3 rgb = vec3(rgba.xyz);

        vec3 yuv = cm * rgb + co;
        yuv = clamp(yuv, 0.f, 1.f);

        imageStore(out_y, pos, vec4(yuv.x));
        imageStore(out_u, pos, vec4(yuv.y));
        imageStore(out_v, pos, vec4(yuv.z));
    }
}
)";

static constexpr char yuv444p_to_rgba[] = R"(
layout (location = 0) uniform mediump sampler2D in_y;
layout (location = 1) uniform mediump sampler2D in_u;
layout (location = 2) uniform mediump sampler2D in_v;
layout (rgba8, binding = 3) writeonly uniform mediump image2D out_img;
layout (location = 4) uniform ivec2 in_img_size;

uniform mat3 cm;
uniform vec3 co;

void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);   // y's width/height
    int width = in_img_size.x;
    int height = in_img_size.y;

    if (pos.x < width && pos.y < height) {
        float y = texelFetch(in_y, pos, 0).x;
        float u = texelFetch(in_u, pos, 0).x;
        float v = texelFetch(in_v, pos, 0).x;

        vec3 yuv = vec3(y, u, v);
        vec3 rgb = cm * (yuv + co);
        rgb = clamp(rgb, 0.f, 1.f);

        imageStore(out_img, pos, vec4(rgb, 1.f));
    }
}
)";

static constexpr char rgba_to_yuv420p[] = R"(
layout (location = 0) uniform mediump sampler2D in_img; //rgba8
layout (rgba8, binding = 1) writeonly uniform mediump image2D out_y;
layout (rgba8, binding = 2) writeonly uniform mediump image2D out_u;
layout (rgba8, binding = 3) writeonly uniform mediump image2D out_v;
layout (location=4) uniform ivec2 in_img_size;

uniform mat3 cm;
uniform vec3 co;

void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    ivec2 uv_hw = in_img_size >> 1;
    int uv_width = uv_hw.x;
    int uv_height = uv_hw.y;

    if (pos.x < uv_width && pos.y < uv_height) {
        vec4 rgba0 = texelFetch(in_img, ivec2(2 * pos.x + 0, 2 * pos.y + 0), 0);
        vec4 rgba1 = texelFetch(in_img, ivec2(2 * pos.x + 1, 2 * pos.y + 0), 0);
        vec4 rgba2 = texelFetch(in_img, ivec2(2 * pos.x + 0, 2 * pos.y + 1), 0);
        vec4 rgba3 = texelFetch(in_img, ivec2(2 * pos.x + 1, 2 * pos.y + 1), 0);

        vec3 rgb0 = vec3(rgba0.xyz);
        vec3 rgb1 = vec3(rgba1.xyz);
        vec3 rgb2 = vec3(rgba2.xyz);
        vec3 rgb3 = vec3(rgba3.xyz);

        vec3 yuv0 = cm * rgb0 + co;
        vec3 yuv1 = cm * rgb1 + co;
        vec3 yuv2 = cm * rgb2 + co;
        vec3 yuv3 = cm * rgb3 + co;

        vec2 uv = (yuv0.yz + yuv1.yz + yuv2.yz + yuv3.yz) / 4.f;
        // uv = uv >> 2;
        vec4 y = vec4(yuv0.x, yuv1.x, yuv2.x, yuv3.x);

        y = clamp(y, 0.f, 1.f);
        uv = clamp(uv, 0.f, 1.f);

        imageStore(out_y, ivec2(2 * pos.x + 0, 2 * pos.y + 0), vec4(y.x));
        imageStore(out_y, ivec2(2 * pos.x + 1, 2 * pos.y + 0), vec4(y.y));
        imageStore(out_y, ivec2(2 * pos.x + 0, 2 * pos.y + 1), vec4(y.z));
        imageStore(out_y, ivec2(2 * pos.x + 1, 2 * pos.y + 1), vec4(y.w));

        imageStore(out_u, pos, vec4(uv.x));
        imageStore(out_v, pos, vec4(uv.y));
    }
}
)";

static constexpr char yuv420p_to_rgba[] = R"(
layout (location = 0) uniform mediump sampler2D in_y; // rgba8
layout (location = 1) uniform mediump sampler2D in_u; // rgba8
layout (location = 2) uniform mediump sampler2D in_v; // rgba8
layout (rgba8, binding = 3) writeonly uniform mediump image2D out_img;
layout (location = 4) uniform ivec2 in_y_size; 

uniform mat3 cm;
uniform vec3 co;

void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);   // uv's width/height
    ivec2 uv_size = in_y_size >> 1;
    int width = uv_size.x;
    int height = uv_size.y;

    if (pos.x < width && pos.y < height) {
        float y0 = texelFetch(in_y, ivec2(pos.x * 2 + 0, pos.y * 2 + 0), 0).x;
        float y1 = texelFetch(in_y, ivec2(pos.x * 2 + 1, pos.y * 2 + 0), 0).x;
        float y2 = texelFetch(in_y, ivec2(pos.x * 2 + 0, pos.y * 2 + 1), 0).x;
        float y3 = texelFetch(in_y, ivec2(pos.x * 2 + 1, pos.y * 2 + 1), 0).x;

        float u = texelFetch(in_u, pos, 0).x;
        float v = texelFetch(in_v, pos, 0).x;

        vec3 yuv0 = vec3(y0, u, v);
        vec3 yuv1 = vec3(y1, u, v);
        vec3 yuv2 = vec3(y2, u, v);
        vec3 yuv3 = vec3(y3, u, v);

        vec3 rgb0 = cm * (yuv0 + co);
        vec3 rgb1 = cm * (yuv1 + co);
        vec3 rgb2 = cm * (yuv2 + co);
        vec3 rgb3 = cm * (yuv3 + co);

        rgb0 = clamp(rgb0, 0.f, 1.f);
        rgb1 = clamp(rgb1, 0.f, 1.f);
        rgb2 = clamp(rgb2, 0.f, 1.f);
        rgb3 = clamp(rgb3, 0.f, 1.f);

        imageStore(out_img, ivec2(pos.x * 2 + 0, pos.y * 2 + 0), vec4(rgb0, 1.f));
        imageStore(out_img, ivec2(pos.x * 2 + 1, pos.y * 2 + 0), vec4(rgb1, 1.f));
        imageStore(out_img, ivec2(pos.x * 2 + 0, pos.y * 2 + 1), vec4(rgb2, 1.f));
        imageStore(out_img, ivec2(pos.x * 2 + 1, pos.y * 2 + 1), vec4(rgb3, 1.f));
    }
}
)";

static constexpr char rgba_to_yuva444[] = R"(
layout (location = 0) uniform mediump sampler2D in_rgba; //rgba
layout (rgba8, binding = 1) writeonly uniform mediump image2D out_yuva;
layout (location = 2) uniform ivec2 in_img_size;

uniform mat3 cm;
uniform vec3 co;

void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    int width = in_img_size.x;
    int height = in_img_size.y;

    if (pos.x < width && pos.y < height) {
        vec4 rgba = texelFetch(in_rgba, pos, 0);
        vec3 rgb = vec3(rgba.xyz);

        vec3 yuv = cm * rgb + co;
        yuv = clamp(yuv, 0.f, 1.f);

        imageStore(out_yuva, pos, vec4(yuv, rgba.w));
    }
}
)";

static constexpr char yuva444_to_rgba[] = R"(
layout (location = 0) uniform mediump sampler2D in_yuva; //yuva
layout (rgba8, binding = 1) writeonly uniform mediump image2D out_img;
layout (location = 2) uniform ivec2 in_img_size;

uniform mat3 cm;
uniform vec3 co;

void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);   // y's width/height
    int width = in_img_size.x;
    int height = in_img_size.y;

    if (pos.x < width && pos.y < height) {
        vec4 yuva = texelFetch(in_yuva, pos, 0);
        vec3 yuv = yuva.xyz;
        vec3 rgb = cm * (yuv + co);
        rgb = clamp(rgb, 0.f, 1.f);

        imageStore(out_img, pos, vec4(rgb, 1.f));
    }
}
)";

static constexpr char rgba_to_hsva[] = R"(
precision mediump float;

layout (location = 0) uniform mediump sampler2D in_rgba; //rgba
layout (rgba8, binding = 1) writeonly uniform mediump image2D out_hsva;
layout (location = 2) uniform ivec2 in_img_size;

vec3 rgb2hsv(vec3 c)
{
    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));

    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    int width = in_img_size.x;
    int height = in_img_size.y;

    if (pos.x < width && pos.y < height) {
        vec4 frgba = texelFetch(in_rgba, pos, 0);
        vec3 hsv = rgb2hsv(frgba.rgb);

        imageStore(out_hsva, pos, vec4(hsv, frgba.w));
    }
}
)";

static constexpr char hsva_to_rgba[] = R"(
precision mediump float;

layout (location = 0) uniform mediump sampler2D in_hsva; //hsva
layout (rgba8, binding = 1) writeonly uniform mediump image2D out_rgba;
layout (location = 2) uniform ivec2 in_img_size;

vec3 hsv2rgb(vec3 c)
{
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    int width = in_img_size.x;
    int height = in_img_size.y;

    if (pos.x < width && pos.y < height) {
        vec4 fhsva = texelFetch(in_hsva, pos, 0);
        vec3 rgb = hsv2rgb(fhsva.xyz);
        imageStore(out_rgba, pos, vec4(rgb, fhsva.w));
    }
}
)";

int Cvt::init(CvtMode mode, const std::string &program_cache_dir) {
    cvt_mode_ = mode;

    std::string program_source;

    switch (cvt_mode_) {
    case CvtMode::RGBA_TO_YUVA444:
        program_source = "rgba_to_yuva444";
        break;
    case CvtMode::RGBA_TO_YUV444P:
        program_source = "rgba_to_yuv444p";
        break;
    case CvtMode::RGBA_TO_YUV420P:
        program_source = "rgba_to_yuv420p";
        break;
    case CvtMode::YUV420P_TO_RGBA:
        program_source = "yuv420p_to_rgba";
        break;
    case CvtMode::YUVA444_TO_RGBA:
        program_source = "yuva444_to_rgba";
        break;
    case CvtMode::YUV444P_TO_RGBA:
        program_source = "yuv444p_to_rgba";
        break;
    case CvtMode::RGBA_TO_HSVA:
        program_source = "rgba_to_hsva";
        break;
    case CvtMode::HSVA_TO_RGBA:
        program_source = "hsva_to_rgba";
        break;
    default:
        OPS_LOG_ERROR("Unknown cvt_mode: %d", cvt_mode_);
        return BMF_LITE_OpsError;
    }

    OPS_CHECK(BMF_LITE_StsOk == GLHelper::instance().build_program(
                                    &program_id_, program_source, "",
                                    program_cache_dir, local_size_, 16, 16, 1),
              "get_program_from_cache_dir fail");

    if (cvt_mode_ != CvtMode::RGBA_TO_HSVA &&
        cvt_mode_ != CvtMode::HSVA_TO_RGBA) {
        cm_loc_ = glGetUniformLocation(program_id_, "cm");
        co_loc_ = glGetUniformLocation(program_id_, "co");
        OPS_CHECK_OPENGL;
    }
    return BMF_LITE_StsOk;
}

int Cvt::run(GLuint in_tex, GLuint out_tex, int width, int height,
             ColorSpace cs, ColorRange in_cr, ColorRange out_cr) {
    if (!(cvt_mode_ == CvtMode::RGBA_TO_YUVA444 ||
          cvt_mode_ == CvtMode::YUVA444_TO_RGBA ||
          cvt_mode_ == CvtMode::RGBA_TO_HSVA ||
          cvt_mode_ == CvtMode::HSVA_TO_RGBA)) {
        OPS_LOG_ERROR("cvt_mode: %d", cvt_mode_);
        return BMF_LITE_OpsError;
    }

    if (width != width_ || height != height_) {
        width_ = width;
        height_ = height;
        auto local_size_x = local_size_[0];
        auto local_size_y = local_size_[1];

        num_groups_x_ = UPDIV(width_, local_size_x);
        num_groups_y_ = UPDIV(height_, local_size_y);
    }
    if ((cvt_mode_ == CvtMode::RGBA_TO_YUVA444 ||
         cvt_mode_ == CvtMode::YUVA444_TO_RGBA) &&
        (cs_ != cs || in_cr_ != in_cr || out_cr_ != out_cr)) {
        cs_ = cs;
        in_cr_ = in_cr;
        out_cr_ = out_cr;

        if (cvt_mode_ == CvtMode::RGBA_TO_YUVA444) {
            if (cs == ColorSpace::BT601 && in_cr == ColorRange::FULL &&
                out_cr == ColorRange::LIMITED) {
                cm_ = cm_rgb2yuv_bt601_f2l;
                co_ = co_rgb2yuv_bt601_f2l;
            } else if (cs == ColorSpace::BT601 &&
                       in_cr == ColorRange::LIMITED &&
                       out_cr == ColorRange::LIMITED) {
                cm_ = cm_rgb2yuv_jpeg;
                co_ = co_rgb2yuv_jpeg;
            } else if (cs == ColorSpace::BT709 && in_cr == ColorRange::FULL &&
                       out_cr == ColorRange::LIMITED) {
                cm_ = cm_rgb2yuv_bt709_f2l;
                co_ = co_rgb2yuv_bt709_f2l;
            } else if (cs == ColorSpace::BT709 &&
                       in_cr == ColorRange::LIMITED &&
                       out_cr == ColorRange::LIMITED) {
                cm_ = cm_rgb2yuv_bt709_l2l;
                co_ = co_rgb2yuv_bt709_l2l;
            } else {
                cm_ = cm_rgb2yuv_jpeg;
                co_ = co_rgb2yuv_jpeg;
            }
        } else {
            if (cs == ColorSpace::BT601 && in_cr == ColorRange::LIMITED &&
                out_cr == ColorRange::FULL) {
                cm_ = cm_yuv2rgb_bt601_l2f;
                co_ = co_yuv2rgb_bt601_l2f;
            } else if (cs == ColorSpace::BT601 &&
                       in_cr == ColorRange::LIMITED &&
                       out_cr == ColorRange::LIMITED) {
                cm_ = cm_yuv2rgb_jpeg;
                co_ = co_yuv2rgb_jpeg;
            } else if (cs == ColorSpace::BT709 &&
                       in_cr == ColorRange::LIMITED &&
                       out_cr == ColorRange::FULL) {
                cm_ = cm_yuv2rgb_bt709_l2f;
                co_ = co_yuv2rgb_bt709_l2f;
            } else if (cs == ColorSpace::BT709 &&
                       in_cr == ColorRange::LIMITED &&
                       out_cr == ColorRange::LIMITED) {
                cm_ = cm_yuv2rgb_bt709_l2l;
                co_ = co_yuv2rgb_bt709_l2l;
            } else {
                cm_ = cm_yuv2rgb_jpeg;
                co_ = co_yuv2rgb_jpeg;
            }
        }
        // set cm/co
        glProgramUniformMatrix3fv(program_id_, cm_loc_, 1, GL_TRUE, cm_);
        glProgramUniform3fv(program_id_, co_loc_, 1, co_);
        OPS_CHECK_OPENGL;
    }

    glUseProgram(program_id_);

    int tex_id = 0;
    glActiveTexture(GL_TEXTURE0 + tex_id);
    glBindTexture(GL_TEXTURE_2D, in_tex);
    glUniform1i(0, tex_id);
    glBindTexture(GL_TEXTURE_2D, in_tex);

    glBindImageTexture(1, out_tex, 0, GL_FALSE, 0, GL_WRITE_ONLY,
                       GL_RGBA8); // out:
    glUniform2i(2, width, height);
    OPS_CHECK_OPENGL;

    glDispatchCompute(num_groups_x_, num_groups_y_, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    OPS_CHECK_OPENGL;
    return BMF_LITE_StsOk;
}

int Cvt::run(GLuint tex0, GLuint tex1, GLuint tex2, GLuint tex3, int width,
             int height, ColorSpace cs, ColorRange in_cr, ColorRange out_cr) {
    if (cvt_mode_ == CvtMode::RGBA_TO_YUV420P ||
        cvt_mode_ == CvtMode::RGBA_TO_YUV444P) {
        return run_1to3(tex0, tex1, tex2, tex3, width, height, cs, in_cr,
                        out_cr);
    } else if (cvt_mode_ == CvtMode::YUV420P_TO_RGBA ||
               cvt_mode_ == CvtMode::YUV444P_TO_RGBA) {
        return run_3to1(tex0, tex1, tex2, tex3, width, height, cs, in_cr,
                        out_cr);
    } else {
        OPS_LOG_ERROR("invalid: mode: %d", cvt_mode_);
        return BMF_LITE_OpsError;
    }
}

int Cvt::run_1to3(GLuint in_tex, GLuint out_y, GLuint out_u, GLuint out_v,
                  int width, int height, ColorSpace cs, ColorRange in_cr,
                  ColorRange out_cr, int in_uv_width, int in_uv_height) {
    if (!(cvt_mode_ == CvtMode::RGBA_TO_YUV420P ||
          cvt_mode_ == CvtMode::RGBA_TO_YUV444P)) {
        OPS_LOG_ERROR("cvt_mode: %d", cvt_mode_);
        return BMF_LITE_OpsError;
    }

    if (width != width_ || height != height_) {
        width_ = width;
        height_ = height;
        auto local_size_x = local_size_[0];
        auto local_size_y = local_size_[1];

        num_groups_x_ = UPDIV(width_, local_size_x);
        num_groups_y_ = UPDIV(height_, local_size_y);
        if (cvt_mode_ == CvtMode::RGBA_TO_YUV420P) {
            num_groups_x_ = UPDIV(width_ >> 1, local_size_x);
            num_groups_y_ = UPDIV(height_ >> 1, local_size_y);
        }
    }
    if (cs_ != cs || in_cr_ != in_cr || out_cr_ != out_cr) {
        cs_ = cs;
        in_cr_ = in_cr;
        out_cr_ = out_cr;

        if (cs == ColorSpace::BT601 && in_cr == ColorRange::FULL &&
            out_cr == ColorRange::LIMITED) {
            cm_ = cm_rgb2yuv_bt601_f2l;
            co_ = co_rgb2yuv_bt601_f2l;
        } else if (cs == ColorSpace::BT601 && in_cr == ColorRange::LIMITED &&
                   out_cr == ColorRange::LIMITED) {
            // cm_ = cm_rgb2yuv_bt601_l2l;
            // co_ = co_rgb2yuv_bt601_l2l;
            cm_ = cm_rgb2yuv_jpeg;
            co_ = co_rgb2yuv_jpeg;
        } else if (cs == ColorSpace::BT709 && in_cr == ColorRange::FULL &&
                   out_cr == ColorRange::LIMITED) {
            cm_ = cm_rgb2yuv_bt709_f2l;
            co_ = co_rgb2yuv_bt709_f2l;
        } else if (cs == ColorSpace::BT709 && in_cr == ColorRange::LIMITED &&
                   out_cr == ColorRange::LIMITED) {
            cm_ = cm_rgb2yuv_bt709_l2l;
            co_ = co_rgb2yuv_bt709_l2l;
        } else {
            cm_ = cm_rgb2yuv_jpeg;
            co_ = co_rgb2yuv_jpeg;
        }

        // set cm/co
        glProgramUniformMatrix3fv(program_id_, cm_loc_, 1, GL_TRUE, cm_);
        OPS_CHECK_OPENGL;
        glProgramUniform3fv(program_id_, co_loc_, 1, co_);
        OPS_CHECK_OPENGL;
    }

    glUseProgram(program_id_);

    int tex_id = 0;
    glActiveTexture(GL_TEXTURE0 + tex_id);
    glBindTexture(GL_TEXTURE_2D, in_tex);
    glUniform1i(0, tex_id);
    OPS_CHECK_OPENGL;

    glBindImageTexture(1, out_y, 0, GL_FALSE, 0, GL_WRITE_ONLY,
                       GL_RGBA8); // out: y
    glBindImageTexture(2, out_u, 0, GL_FALSE, 0, GL_WRITE_ONLY,
                       GL_RGBA8); // out: u
    glBindImageTexture(3, out_v, 0, GL_FALSE, 0, GL_WRITE_ONLY,
                       GL_RGBA8); // out: v

    glUniform2i(4, width, height);

    glDispatchCompute(num_groups_x_, num_groups_y_, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    OPS_CHECK_OPENGL;
    return BMF_LITE_StsOk;
}

int Cvt::run_3to1(GLuint in_y, GLuint in_u, GLuint in_v, GLuint out_tex,
                  int width, int height, ColorSpace cs, ColorRange in_cr,
                  ColorRange out_cr, int in_uv_width, int in_uv_height) {
    if (!(cvt_mode_ == CvtMode::YUV444P_TO_RGBA ||
          cvt_mode_ == CvtMode::YUV420P_TO_RGBA)) {
        OPS_LOG_ERROR("cvt_mode: %d", cvt_mode_);
        return BMF_LITE_OpsError;
    }

    if (width != width_ || height != height_) {
        width_ = width;
        height_ = height;
        auto local_size_x = local_size_[0];
        auto local_size_y = local_size_[1];

        num_groups_x_ = UPDIV(width_, local_size_x);
        num_groups_y_ = UPDIV(height_, local_size_y);
        if (cvt_mode_ == CvtMode::YUV420P_TO_RGBA) {
            num_groups_x_ = UPDIV(width_ >> 1, local_size_x);
            num_groups_y_ = UPDIV(height_ >> 1, local_size_y);
        }
    }
    if (cs_ != cs || in_cr_ != in_cr || out_cr_ != out_cr) {
        cs_ = cs;
        in_cr_ = in_cr;
        out_cr_ = out_cr;
        if (cs == ColorSpace::BT601 && in_cr == ColorRange::LIMITED &&
            out_cr == ColorRange::FULL) {
            cm_ = cm_yuv2rgb_bt601_l2f;
            co_ = co_yuv2rgb_bt601_l2f;
        } else if (cs == ColorSpace::BT601 && in_cr == ColorRange::LIMITED &&
                   out_cr == ColorRange::LIMITED) {
            cm_ = cm_yuv2rgb_jpeg;
            co_ = co_yuv2rgb_jpeg;
        } else if (cs == ColorSpace::BT709 && in_cr == ColorRange::LIMITED &&
                   out_cr == ColorRange::FULL) {
            cm_ = cm_yuv2rgb_bt709_l2f;
            co_ = co_yuv2rgb_bt709_l2f;
        } else if (cs == ColorSpace::BT709 && in_cr == ColorRange::LIMITED &&
                   out_cr == ColorRange::LIMITED) {
            cm_ = cm_yuv2rgb_bt709_l2l;
            co_ = co_yuv2rgb_bt709_l2l;
        } else {
            cm_ = cm_yuv2rgb_jpeg;
            co_ = co_yuv2rgb_jpeg;
        }
        // set cm/co
        glProgramUniformMatrix3fv(program_id_, cm_loc_, 1, GL_TRUE, cm_);
        OPS_CHECK_OPENGL;
        glProgramUniform3fv(program_id_, co_loc_, 1, co_);
        OPS_CHECK_OPENGL;
    }

    glUseProgram(program_id_);

    if (cvt_mode_ == CvtMode::YUV420P_TO_RGBA ||
        cvt_mode_ == CvtMode::YUV444P_TO_RGBA) {
        int tex_id = 0;
        glActiveTexture(GL_TEXTURE0 + tex_id);
        glBindTexture(GL_TEXTURE_2D, in_y);
        glUniform1i(0, tex_id);
        OPS_CHECK_OPENGL;

        tex_id++;
        glActiveTexture(GL_TEXTURE0 + tex_id);
        glBindTexture(GL_TEXTURE_2D, in_u);
        glUniform1i(1, tex_id);
        OPS_CHECK_OPENGL;

        tex_id++;
        glActiveTexture(GL_TEXTURE0 + tex_id);
        glBindTexture(GL_TEXTURE_2D, in_v);
        glUniform1i(2, tex_id);
        OPS_CHECK_OPENGL;

        glBindImageTexture(3, out_tex, 0, GL_FALSE, 0, GL_WRITE_ONLY,
                           GL_RGBA8); // out: rgba
        OPS_CHECK_OPENGL;

        glUniform2i(4, width, height);
        OPS_CHECK_OPENGL;
    }
    glDispatchCompute(num_groups_x_, num_groups_y_, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    OPS_CHECK_OPENGL;
    return BMF_LITE_StsOk;
}

Cvt::~Cvt() {
    if (program_id_ != GL_NONE) {
        glDeleteProgram(program_id_);
        program_id_ = GL_NONE;
    }
}

} // namespace opengl
} // namespace bmf_lite
