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
#include "opengl/denoise/denoise.h"
namespace bmf_lite {
namespace opengl {
static constexpr char denoise_cs[] = R"(
precision mediump float;

layout (rgba8, binding = 0) writeonly uniform mediump image2D out_rgba; 
layout (rgba8, binding = 1) writeonly uniform mediump image2D cur_rgba; 
layout (location = 0) uniform mediump sampler2D in_rgba; // rgba8
#ifdef USE_TEMPORAL_MIX
layout (location = 1) uniform mediump sampler2D pre_rgba_tex; // rgba8
#define TEMPORAL_SLOPE 15.0
#endif
layout (location = 2) uniform ivec2 in_size;

float get_luma(vec4 rgba) {
	return dot(vec3(0.299, 0.587, 0.114), rgba.rgb);
}

float gaussian(float x, float s, float m) {
    float scaled = (x - m) / s;
    return exp(-0.5 * scaled * scaled);
}

void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    if (pos.x >= in_size.x || pos.y >= in_size.y) {
        return;
    }
    vec4 histogram_v[5][5];
    float n = 0.0;
    vec4 sum = vec4(0.0);

    float is = 0.05;
    float ss = 1.0;

    int x_m_1 = max(pos.x - 1, 0);
    int x_a_1 = min(pos.x + 1, in_size.x - 1);
    int x_m_2 = max(pos.x - 2, 0);
    int x_a_2 = min(pos.x + 2, in_size.x - 1);

    int y_m_1 = max(pos.y - 1, 0);
    int y_a_1 = min(pos.y + 1, in_size.y - 1);
    int y_m_2 = max(pos.y - 2, 0);
    int y_a_2 = min(pos.y + 2, in_size.y - 1);

    histogram_v[0][0] = texelFetch(in_rgba, ivec2(x_m_2, y_m_2), 0);
    histogram_v[1][0] = texelFetch(in_rgba, ivec2(x_m_1, y_m_2), 0);
    histogram_v[2][0] = texelFetch(in_rgba, ivec2(pos.x, y_m_2), 0);
    histogram_v[3][0] = texelFetch(in_rgba, ivec2(x_a_1, y_m_2), 0);
    histogram_v[4][0] = texelFetch(in_rgba, ivec2(x_a_2, y_m_2), 0);

    histogram_v[0][1] = texelFetch(in_rgba, ivec2(x_m_2, y_m_1), 0);
    histogram_v[1][1] = texelFetch(in_rgba, ivec2(x_m_1, y_m_1), 0);
    histogram_v[2][1] = texelFetch(in_rgba, ivec2(pos.x, y_m_1), 0);
    histogram_v[3][1] = texelFetch(in_rgba, ivec2(x_a_1, y_m_1), 0);
    histogram_v[4][1] = texelFetch(in_rgba, ivec2(x_a_2, y_m_1), 0);

    histogram_v[0][2] = texelFetch(in_rgba, ivec2(x_m_2, pos.y), 0);
    histogram_v[1][2] = texelFetch(in_rgba, ivec2(x_m_1, pos.y), 0);
    histogram_v[2][2] = texelFetch(in_rgba, ivec2(pos.x, pos.y), 0);
    histogram_v[3][2] = texelFetch(in_rgba, ivec2(x_a_1, pos.y), 0);
    histogram_v[4][2] = texelFetch(in_rgba, ivec2(x_a_2, pos.y), 0);

    histogram_v[0][3] = texelFetch(in_rgba, ivec2(x_m_2, y_a_1), 0);
    histogram_v[1][3] = texelFetch(in_rgba, ivec2(x_m_1, y_a_1), 0);
    histogram_v[2][3] = texelFetch(in_rgba, ivec2(pos.x, y_a_1), 0);
    histogram_v[3][3] = texelFetch(in_rgba, ivec2(x_a_1, y_a_1), 0);
    histogram_v[4][3] = texelFetch(in_rgba, ivec2(x_a_2, y_a_1), 0);

    histogram_v[0][4] = texelFetch(in_rgba, ivec2(x_m_2, y_a_2), 0);
    histogram_v[1][4] = texelFetch(in_rgba, ivec2(x_m_1, y_a_2), 0);
    histogram_v[2][4] = texelFetch(in_rgba, ivec2(pos.x, y_a_2), 0);
    histogram_v[3][4] = texelFetch(in_rgba, ivec2(x_a_1, y_a_2), 0);
    histogram_v[4][4] = texelFetch(in_rgba, ivec2(x_a_2, y_a_2), 0);

    float vc = get_luma(histogram_v[2][2]);

    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            float w = gaussian(get_luma(histogram_v[i][j]), is, vc) * gaussian(length(vec2(i - 2, j - 2)), ss, 0.0);
            n += w;
            sum += histogram_v[i][j] * w;
        }
    }
    vec4 result = sum / n;
#ifdef USE_TEMPORAL_MIX
    float result_y = get_luma(result);
    vec4 pre_rgba = texelFetch(pre_rgba_tex, ivec2(pos.x, pos.y), 0);
    float pre_y = get_luma(pre_rgba);
    float temporal_weight = max(min(abs(result_y - pre_y) * TEMPORAL_SLOPE, 1.0), 0.0);
    result = mix((pre_rgba + result) * 0.5f, result, temporal_weight);
#endif
    imageStore(out_rgba, pos, result);
    imageStore(cur_rgba, pos, result);
}
)";

int Denoise::init(const std::string &program_cache_dir) {
    OPS_CHECK(inited_ == false, "already inited");

    GLint local_size[3];

    OPS_CHECK(BMF_LITE_StsOk == GLHelper::instance().build_program(
                                    &program_, denoise_cs,
                                    "#define USE_TEMPORAL_MIX 1\n",
                                    program_cache_dir, local_size, 16, 16, 1),
              "compile program denoise error");
    OPS_CHECK(BMF_LITE_StsOk == GLHelper::instance().build_program(
                                    &program_no_mix_, denoise_cs, "",
                                    program_cache_dir, local_size, 16, 16, 1),
              "compile program denoise_no_mix error");

    local_size_x_ = local_size[0];
    local_size_y_ = local_size[1];

    inited_ = true;
    return BMF_LITE_StsOk;
}

int Denoise::run(GLuint in_tex, GLuint out_tex, int width, int height) {
    OPS_CHECK(inited_, "please init first");
    if (in_width_ != width || in_height_ != height) {
        in_width_ = width;
        in_height_ = height;

        if (tex_ping_ != GL_NONE) {
            glDeleteTextures(1, &tex_ping_);
            tex_ping_ = GL_NONE;
        }
        tex_ping_ = GLHelper::instance().gen_itex(width, height, GL_RGBA8,
                                                  GL_TEXTURE_2D, GL_NEAREST);

        if (tex_pong_ != GL_NONE) {
            glDeleteTextures(1, &tex_pong_);
            tex_pong_ = GL_NONE;
        }
        tex_pong_ = GLHelper::instance().gen_itex(width, height, GL_RGBA8,
                                                  GL_TEXTURE_2D, GL_NEAREST);
        first_run_ = true;
    }
    GLint groups_x0 = UPDIV(width, local_size_x_);
    GLint groups_y0 = UPDIV(height, local_size_y_);

    if (first_run_ == true) {
        glUseProgram(program_no_mix_);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, in_tex);
        glUniform1i(0, 0);

        glUniform2i(2, width, height);

        glBindImageTexture(0, out_tex, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8);
        glBindImageTexture(1, tex_pong_, 0, GL_FALSE, 0, GL_WRITE_ONLY,
                           GL_RGBA8);

        glDispatchCompute(groups_x0, groups_y0, 1);
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT |
                        GL_TEXTURE_FETCH_BARRIER_BIT);
        first_run_ = false;
    } else {
        glUseProgram(program_);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, in_tex);
        glUniform1i(0, 0);

        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, tex_ping_);
        glUniform1i(1, 1);

        glUniform2i(2, width, height);

        glBindImageTexture(0, out_tex, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8);
        glBindImageTexture(1, tex_pong_, 0, GL_FALSE, 0, GL_WRITE_ONLY,
                           GL_RGBA8);

        glDispatchCompute(groups_x0, groups_y0, 1);
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT |
                        GL_TEXTURE_FETCH_BARRIER_BIT);
    }

    // swap tex_ping_ and tex_pong_
    GLuint tmp = tex_ping_;
    tex_ping_ = tex_pong_;
    tex_pong_ = tmp;

    return 0;
}

Denoise::~Denoise() {
    if (program_ != GL_NONE) {
        glDeleteProgram(program_);
        program_ = GL_NONE;
    }
    if (program_no_mix_ != GL_NONE) {
        glDeleteProgram(program_no_mix_);
        program_no_mix_ = GL_NONE;
    }
    if (tex_ping_ != GL_NONE) {
        glDeleteTextures(1, &tex_ping_);
        tex_ping_ = GL_NONE;
    }
    if (tex_pong_ != GL_NONE) {
        glDeleteTextures(1, &tex_pong_);
        tex_pong_ = GL_NONE;
    }
}
} // namespace opengl
} // namespace bmf_lite
