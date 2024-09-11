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
#include "resize.h"

namespace bmf_lite {
namespace opengl {
static constexpr char resize_src[] = R"(
precision mediump float;
layout (location = 0) uniform mediump sampler2D in_img; 
layout (rgba8, binding = 1) writeonly uniform mediump image2D out_img; 
layout (location=2) uniform ivec2 in_img_size;
layout (location=3) uniform ivec2 out_img_size;
void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    int dst_width = out_img_size.x;
    int dst_height = out_img_size.y;
    int src_width = in_img_size.x;
    int src_height = in_img_size.y;
    if (pos.x < dst_width && pos.y < dst_height) {
        float scale_x =  float(src_width) / float(dst_width);
        float scale_y = float(src_height) / float(dst_height);
        float raw_in_x = (float(pos.x)+0.5f)*scale_x-0.5f;
        float raw_in_y = (float(pos.y)+0.5f)*scale_y-0.5f;
        int in_x = int(raw_in_x);
        int in_y = int(raw_in_y);
        float u = raw_in_x - float(in_x);
        float v = raw_in_y - float(in_y);
        if(in_x<0) {
            in_x = 0;
            u = 0.f;
        }
        if(in_y<0) {
            in_y = 0;
            v = 0.f;
        }
        if (in_x + 1 < src_width && in_y + 1 < src_height) {
            vec4 pixel_one = texelFetch(in_img, ivec2(in_x, in_y), 0);
            vec4 pixel_two = texelFetch(in_img, ivec2(in_x + 1, in_y), 0);
            vec4 pixel_three = texelFetch(in_img, ivec2(in_x, in_y + 1), 0);
            vec4 pixel_four = texelFetch(in_img, ivec2(in_x + 1, in_y + 1), 0);

            vec3 rgb0 = vec3(pixel_one.xyz);
            vec3 rgb1 = vec3(pixel_two.xyz);
            vec3 rgb2 = vec3(pixel_three.xyz);
            vec3 rgb3 = vec3(pixel_four.xyz);
            vec3 pixel_rgb = (1.f - u) * (1.f - v) * rgb0 + (1.f - v) * u * rgb1  + (1.f - u) * v * rgb2  + u * v *rgb3 ;
            vec3 rgb = clamp(vec3(pixel_rgb), 0.f, 255.f);
            imageStore(out_img, pos, vec4(rgb, 255.f));
        } else {
            vec4 pixel_rgb =  texelFetch(in_img, ivec2(in_x, in_y), 0);
            vec3 rgb = clamp(vec3(pixel_rgb.xyz), 0.f, 255.f);
            imageStore(out_img, pos, vec4(rgb, 255.f));
        }
    }
}
)";

int Resize::init(const std::string &program_cache_dir) {
    std::string program_source = resize_src;
    OPS_CHECK(BMF_LITE_StsOk == GLHelper::instance().build_program(
                                    &program_id_, program_source, "",
                                    program_cache_dir, local_size_, 16, 16, 1),
              "get_program_from_cache_dir fail");
    OPS_CHECK_OPENGL;
    inited_ = true;
    return BMF_LITE_StsOk;
}

int Resize::run(GLuint in_rgba, GLuint out_rgba, int in_width, int in_height,
                int out_width, int out_height) {
    OPS_CHECK(inited_, "init first");
    if (out_width != width_ || out_height != height_) {
        width_ = out_width;
        height_ = out_height;
        auto local_size_x = local_size_[0];
        auto local_size_y = local_size_[1];

        num_groups_x_ = UPDIV(width_, local_size_x);
        num_groups_y_ = UPDIV(height_, local_size_y);
    }
    glUseProgram(program_id_);
    if (in_width == 0 || in_height == 0) {
        OPS_LOG_DEBUG("in_width: %d, in_height: %d\n", in_width, in_height);
        return BMF_LITE_StsBadArg;
    }
    int tex_id = 0;
    glActiveTexture(GL_TEXTURE0 + tex_id);
    glBindTexture(GL_TEXTURE_2D, in_rgba);
    glUniform1i(0, tex_id);
    OPS_CHECK_OPENGL;
    glBindImageTexture(1, out_rgba, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8);
    OPS_CHECK_OPENGL;
    glUniform2i(2, in_width, in_height);
    glUniform2i(3, out_width, out_height);
    OPS_CHECK_OPENGL;
    glDispatchCompute(num_groups_x_, num_groups_y_, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    OPS_CHECK_OPENGL;
    return BMF_LITE_StsOk;
}

Resize::~Resize() {
    if (program_id_ != GL_NONE) {
        glDeleteProgram(program_id_);
        program_id_ = GL_NONE;
    }
}
} // namespace opengl
} // namespace bmf_lite
