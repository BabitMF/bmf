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
#include "arithmetic.h"

namespace bmf_lite {
namespace opengl {
static constexpr char arithmetic_src[] = R"(
precision mediump float;
precision mediump int;
layout (location = 0) uniform mediump sampler2D in_tex0;  
layout (location = 1)  uniform mediump sampler2D in_tex1;  
layout (rgba8, binding = 0) writeonly uniform mediump image2D out_img; 
layout (location = 2) uniform ivec2 w_h;
layout (location = 3) uniform vec3 params; //(a, b, c)
void main() { 
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    if (pos.x < w_h.x && pos.y < w_h.y) {
        vec4 data0 = texelFetch(in_tex0, pos, 0);
        vec4 data1 = texelFetch(in_tex1, pos, 0);
        vec4 out_data = params.x * data0 + params.y * data1 + params.z;
        out_data = clamp(out_data, 0.f, 1.f);
        imageStore(out_img, pos, out_data);
    }
}
)";

int Arithmetic::init(const std::string &program_cache_dir) {
    OPS_CHECK(!inited_, "already inited");
    std::string program_source = arithmetic_src;

    OPS_CHECK(BMF_LITE_StsOk == GLHelper::instance().build_program(
                                    &program_id_, program_source, "",
                                    program_cache_dir, local_size_, 16, 16, 1),
              "get_program_from_cache_dir fail");
    inited_ = true;
    return BMF_LITE_StsOk;
}

int Arithmetic::run(GLuint in_tex0, GLuint in_tex1, GLuint out_tex, int width,
                    int height, float a, float b, float c) {
    OPS_CHECK(inited_, "init first");
    auto num_groups_x = UPDIV(width, local_size_[0]);
    auto num_groups_y = UPDIV(height, local_size_[1]);

    glUseProgram(program_id_);

    int tex_id = 0;
    glActiveTexture(GL_TEXTURE0 + tex_id);
    glBindTexture(GL_TEXTURE_2D, in_tex0);
    glUniform1i(0, tex_id);

    tex_id++;
    glActiveTexture(GL_TEXTURE0 + tex_id);
    glBindTexture(GL_TEXTURE_2D, in_tex1);
    glUniform1i(1, tex_id);

    glBindImageTexture(0, out_tex, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8);

    glUniform2i(2, width, height);
    glUniform3f(3, a, b, c);

    glDispatchCompute(num_groups_x, num_groups_y, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    OPS_CHECK_OPENGL;
    return BMF_LITE_StsOk;
}

Arithmetic::~Arithmetic() {
    if (program_id_ != GL_NONE) {
        glDeleteProgram(program_id_);
        program_id_ = GL_NONE;
    }
}
} // namespace opengl
} // namespace bmf_lite
