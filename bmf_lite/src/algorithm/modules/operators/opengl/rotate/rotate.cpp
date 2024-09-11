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
#include "rotate.h"

namespace bmf_lite {
namespace opengl {
static constexpr char rotate_src[] = R"(
precision mediump float;
precision mediump int;

layout (location = 0) uniform mediump sampler2D in_tex;  
layout (rgba8, binding = 0) writeonly uniform mediump image2D out_img; 
layout (location = 1) uniform ivec2 w_h;
layout (location = 2) uniform int degree;
void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    if (pos.x < w_h.x && pos.y < w_h.y) {
        vec4 rgba = texelFetch(in_tex, pos, 0);
        ivec2 dst_pos = pos;
        if (degree == 90) {
            dst_pos.x = w_h.y - pos.y - 1;
            dst_pos.y = pos.x;
        } else if (degree == 180) {
            dst_pos.x = w_h.x - pos.x - 1;
            dst_pos.y = w_h.y - pos.y - 1;      
        } else if (degree == 270) {
            dst_pos.x = pos.y;
            dst_pos.y = w_h.x - pos.x - 1;
        }
        imageStore(out_img, dst_pos, rgba);
    }
})";

int Rotate::init(const std::string &program_cache_dir) {
    OPS_CHECK(!inited_, "already inited");
    std::string program_source = rotate_src;
    OPS_CHECK(BMF_LITE_StsOk == GLHelper::instance().build_program(
                                    &program_id_, program_source, "",
                                    program_cache_dir, local_size_, 16, 16, 1),
              "get_program_from_cache_dir fail");
    OPS_CHECK_OPENGL;

    inited_ = true;
    return BMF_LITE_StsOk;
}

int Rotate::run(GLuint in_tex, GLuint out_tex, int width, int height,
                int angle) {
    OPS_CHECK(inited_, "init first");
    auto num_groups_x = UPDIV(width, local_size_[0]);
    auto num_groups_y = UPDIV(height, local_size_[1]);

    glUseProgram(program_id_);

    int tex_id = 0;
    glActiveTexture(GL_TEXTURE0 + tex_id);
    glBindTexture(GL_TEXTURE_2D, in_tex);
    glUniform1i(0, tex_id);

    glBindImageTexture(0, out_tex, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8);

    glUniform2i(1, width, height);
    glUniform1i(2, angle);

    glDispatchCompute(num_groups_x, num_groups_y, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    return BMF_LITE_StsOk;
}

Rotate::~Rotate() {
    if (program_id_ != GL_NONE) {
        glDeleteProgram(program_id_);
        program_id_ = GL_NONE;
    }
}
} // namespace opengl
} // namespace bmf_lite
