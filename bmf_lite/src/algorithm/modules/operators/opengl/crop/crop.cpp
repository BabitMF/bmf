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
#include "crop.h"

namespace bmf_lite {
namespace opengl {
static constexpr char crop_src[] = R"(
precision mediump float;
precision mediump int;

layout (location = 0) uniform mediump sampler2D in_tex;  
layout (rgba8, binding = 0) writeonly uniform mediump image2D out_img; 
layout (location = 1) uniform ivec4 params; //(x, y, ow, oh)
void main() { 
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    int ow = params.z;
    int oh = params.w;
    if (pos.x < ow && pos.y < oh) {
        vec4 data = texelFetch(in_tex, pos + params.xy, 0);
        imageStore(out_img, pos, data);
    }
})";

int Crop::init(const std::string &program_cache_dir) {
    OPS_CHECK(!inited_, "already inited");
    std::string program_source = crop_src;
    OPS_CHECK(BMF_LITE_StsOk == GLHelper::instance().build_program(
                                    &program_id_, program_source, "",
                                    program_cache_dir, local_size_, 16, 16, 1),
              "get_program_from_cache_dir fail");
    inited_ = true;
    return BMF_LITE_StsOk;
}

int Crop::run(GLuint in_tex, GLuint out_tex, int in_width, int in_height,
              int start_x, int start_y, int out_width, int out_height) {
    OPS_CHECK(inited_, "init first");

    OPS_CHECK(start_x >= 0 && start_x < in_width, "invalid start_x: %d",
              start_x);
    OPS_CHECK(start_y >= 0 && start_y < in_height, "invalid start_y: %d",
              start_y);

    if (out_width + start_x > in_width) {
        out_width = in_width - start_x;
    }
    if (out_height + start_y > in_height) {
        out_height = in_height - start_y;
    }

    auto num_groups_x = UPDIV(out_width, local_size_[0]);
    auto num_groups_y = UPDIV(out_height, local_size_[1]);

    glUseProgram(program_id_);

    int tex_id = 0;
    glActiveTexture(GL_TEXTURE0 + tex_id);
    glBindTexture(GL_TEXTURE_2D, in_tex);
    glUniform1i(0, tex_id);

    glBindImageTexture(0, out_tex, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8);
    glUniform4i(1, start_x, start_y, out_width, out_height);

    glDispatchCompute(num_groups_x, num_groups_y, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    return BMF_LITE_StsOk;
}

Crop::~Crop() {
    if (program_id_ != GL_NONE) {
        glDeleteProgram(program_id_);
        program_id_ = GL_NONE;
    }
}
} // namespace opengl
} // namespace bmf_lite
