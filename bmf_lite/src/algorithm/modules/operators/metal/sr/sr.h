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
#ifndef _BMF_ALGORITHM_MODULES_OPS_METAL_SR_H_
#define _BMF_ALGORITHM_MODULES_OPS_METAL_SR_H_
#include <map>
#include <vector>
#include "metal/metal_helper.h"
namespace bmf_lite {
namespace metal {
class Sr {
  public:
    int init();
    /**
     * @brief
     *
     * @param in_tex input rgba texture
     * @param out_tex output rgba texture
     * @param command_queue
     * @return true
     * @return false
     */
    int run(id<MTLTexture> in_tex, id<MTLTexture> out_tex,
            id<MTLCommandQueue> command_queue);
    int run(id<MTLTexture> in_tex_y, id<MTLTexture> in_tex_uv,
            id<MTLTexture> out_tex_y, id<MTLTexture> out_tex_uv,
            id<MTLCommandQueue> command_queue);

  private:
    id<MTLComputePipelineState> cps0_ = nil;
    id<MTLComputePipelineState> cps1_ = nil;
    id<MTLComputePipelineState> cps2_ = nil;
    id<MTLComputePipelineState> cps3_ = nil;
    id<MTLComputePipelineState> cps4_ = nil;
    id<MTLComputePipelineState> rgba_to_nv12_ = nil;
    id<MTLComputePipelineState> nv12_to_rgba_ = nil;

    MTLSize group_size_;

    id<MTLTexture> tex0_;
    id<MTLTexture> tex1_;

    id<MTLTexture> tex_rgba_in_;
    id<MTLTexture> tex_rgba_out_;

    int in_width_ = 0;
    int in_height_ = 0;
    int in_width_nv12_ = 0;
    int in_height_nv12_ = 0;

    bool inited_ = false;
}; // class Sr

} // namespace metal
} // namespace bmf_lite

#endif
