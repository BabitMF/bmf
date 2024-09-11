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
#include "statistic.h"

namespace bmf_lite {
namespace opengl {
static constexpr char statistic_src[] = R"(
layout (location = 0) uniform mediump sampler2D in_rgba;
layout (location = 1) uniform ivec2 in_size;
layout(binding = 0) buffer FEATURE     
{   
    uint histograms[256];
    uint min;
    uint max;
    uint sum;
} features;
shared uint shared_sum[256];
shared uint shared_max[256];
shared uint shared_min[256];
void main()
{   
    uint tmp_max = uint(0);
    uint tmp_sum = uint(0);
    uint tmp_min = uint(1000);
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    uint localID = gl_LocalInvocationIndex;
    uint localSize = gl_WorkGroupSize.x * gl_WorkGroupSize.y;
    int height = in_size.y;
    int width = in_size.x;
    for(int ih = pos.y; ih < height; ih += 256){
        for(int iw = pos.x; iw < width; iw += 256){
            ivec2 pos_in = ivec2(iw, ih);
            if(pos_in.x < width && pos_in.y < height){
                uint y = clamp(uint(round(255.f * dot(vec3(0.299, 0.587, 0.114), texelFetch(in_rgba, pos_in, 0).xyz))), uint(0), uint(255));
                tmp_min = min(tmp_min, y);
                tmp_max = max(tmp_max, y);
                tmp_sum += y;
                atomicAdd(features.histograms[y], uint(1));
            }
        }
    }
    shared_max[localID] = tmp_max;
    shared_sum[localID] = tmp_sum;
    shared_min[localID] = tmp_min;
    for(uint offset = (localSize / uint(2)); offset > uint(0); offset >>= 1) {
        memoryBarrier();
        barrier();
        if(localID < offset) {
            shared_sum[localID] += shared_sum[localID + offset];
            shared_min[localID] = min(shared_min[localID], shared_min[localID + offset]);
            shared_max[localID] = max(shared_max[localID], shared_max[localID + offset]);
        }
    }
    memoryBarrier();
    barrier();
    if(localID == uint(0)){
        atomicAdd(features.sum, shared_sum[0]);
        atomicMax(features.max, shared_max[0]);
        atomicMin(features.min, shared_min[0]);
    }
})";

int Statistic::init(const std::string &program_cache_dir) {
    OPS_CHECK(!inited_, "already inited");
    std::string program_source = statistic_src;

    OPS_CHECK(BMF_LITE_StsOk == GLHelper::instance().build_program(
                                    &program_id_, program_source, "",
                                    program_cache_dir, local_size_, 16, 16, 1),
              "get_program_from_cache_dir fail");

    glGenBuffers(1, &features_);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, features_);
    int tmp_data[259] = {};
    memset(tmp_data, 0, 259 * sizeof(int));
    glBufferData(GL_SHADER_STORAGE_BUFFER, 259 * sizeof(int), tmp_data,
                 GL_DYNAMIC_READ);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    inited_ = true;
    return BMF_LITE_StsOk;
}

int Statistic::run(GLuint in_tex, int width, int height, int *histogram,
                   int *min, int *max, int *sum) {
    OPS_CHECK(inited_, "init first");
    OPS_CHECK(histogram != nullptr && min != nullptr && max != nullptr &&
                  sum != nullptr,
              "histogram, min, max, sum must not be null");
    auto num_groups_x = UPDIV(256, local_size_[0]);
    auto num_groups_y = UPDIV(256, local_size_[1]);

    glUseProgram(program_id_);

    int tex_id = 0;
    glActiveTexture(GL_TEXTURE0 + tex_id);
    glBindTexture(GL_TEXTURE_2D, in_tex);
    glUniform1i(0, tex_id);

    glUniform2i(1, width, height);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, features_);
    glDispatchCompute(num_groups_x, num_groups_y, 1);
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT |
                    GL_SHADER_STORAGE_BARRIER_BIT);

    uint32_t *data = (uint32_t *)glMapBufferRange(
        GL_SHADER_STORAGE_BUFFER, 0, 259 * sizeof(int), GL_MAP_READ_BIT);
    OPS_CHECK(data != NULL, "map data buffer error");
    memset(data, 0, 259 * sizeof(int));
    memcpy(histogram, data, 256 * sizeof(int));
    *min = data[256];
    *max = data[257];
    *sum = data[258];
    OPS_CHECK(glUnmapBuffer(GL_SHADER_STORAGE_BUFFER) == GL_TRUE,
              "unmap data erro");
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, 0);

    OPS_CHECK_OPENGL;
    return BMF_LITE_StsOk;
}

Statistic::~Statistic() {
    if (program_id_ != GL_NONE) {
        glDeleteProgram(program_id_);
        program_id_ = GL_NONE;
    }
}
} // namespace opengl
} // namespace bmf_lite
