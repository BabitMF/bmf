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
#include "metal/denoise/denoise.h"
#include "metal/metal_helper.h"
#include <numeric>

namespace bmf_lite {
namespace metal {

int Denoise::init() {
    inited_ = false;

    OPS_CHECK(MetalHelper::instance().support_non_uniform_tg(), "not support non_uniform_tg");

    OPS_CHECK(BMF_LITE_StsOk == MetalHelper::instance().new_compute_pileline_state(&denoise_, @"denoise"), "can not find denoise function");
    OPS_CHECK(BMF_LITE_StsOk == MetalHelper::instance().new_compute_pileline_state(&denoise_no_mix_, @"denoise_no_mix"), "can not find denoise_no_mix function");
    OPS_CHECK(BMF_LITE_StsOk == MetalHelper::instance().new_compute_pileline_state(&denoise_nv12_, @"denoise_nv12"), "can not find denoise_no_mix function");
    OPS_CHECK(BMF_LITE_StsOk == MetalHelper::instance().new_compute_pileline_state(&denoise_nv12_no_mix_, @"denoise_nv12_no_mix"), "can not find denoise_no_mix function");
    group_size_ = MTLSizeMake(16, 16, 1);

    inited_ = true;
    return BMF_LITE_StsOk;
}

int Denoise::run(id<MTLTexture> in_tex, id<MTLTexture> out_tex, id<MTLCommandQueue> command_queue) {
    OPS_CHECK(inited_, "please call init first");
    OPS_CHECK(in_tex != nil, "input tex is nil");
    OPS_CHECK(out_tex != nil, "output tex is nil");

    id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
    OPS_CHECK(command_buffer != nil, "command_buffer is nil");
    id<MTLComputeCommandEncoder> command_encoder = [command_buffer computeCommandEncoder];
    OPS_CHECK(command_encoder != nil, "command_encoder is nil");

    if (in_width_ != in_tex.width || in_height_ != in_tex.height) {
        in_width_ = in_tex.width;
        in_height_ = in_tex.height;

        OPS_CHECK(BMF_LITE_StsOk == MetalHelper::instance().gen_tex(&tex_ping_, MTLPixelFormatRGBA8Unorm, in_width_, in_height_, MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite, MTLStorageModePrivate), 
                        "create tex_ping_ error");
        OPS_CHECK(BMF_LITE_StsOk == MetalHelper::instance().gen_tex(&tex_pong_, MTLPixelFormatRGBA8Unorm, in_width_, in_height_, MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite, MTLStorageModePrivate), 
                        "create tex_pong_ error");
        first_run_ = true;
    }

    if (first_run_ == true) {
        [command_encoder setComputePipelineState:denoise_no_mix_];
        [command_encoder setTexture:in_tex   atIndex:0];
        [command_encoder setTexture:out_tex    atIndex:2];
        [command_encoder setTexture:tex_pong_    atIndex:3];
        [command_encoder dispatchThreads:MTLSizeMake(in_width_, in_height_, 1) threadsPerThreadgroup:group_size_];
        [command_encoder endEncoding];
        [command_buffer commit];
        first_run_ = false;
    } else {
        [command_encoder setComputePipelineState:denoise_];
        [command_encoder setTexture:in_tex   atIndex:0];
        [command_encoder setTexture:tex_ping_   atIndex:1];
        [command_encoder setTexture:out_tex    atIndex:2];
        [command_encoder setTexture:tex_pong_    atIndex:3];
        [command_encoder dispatchThreads:MTLSizeMake(in_width_, in_height_, 1) threadsPerThreadgroup:group_size_];
        [command_encoder endEncoding];
        [command_buffer commit];
    }
    // swap ping/pong
    id<MTLTexture> tmp = tex_ping_;
    tex_ping_ = tex_pong_;
    tex_pong_ = tmp;
   
    return BMF_LITE_StsOk;
}

int Denoise::run(id<MTLTexture> in_tex_y, id<MTLTexture> in_tex_uv, id<MTLTexture> out_tex_y, id<MTLTexture> out_tex_uv, id<MTLCommandQueue> command_queue) {
    OPS_CHECK(inited_, "please call init first");
    OPS_CHECK(in_tex_y != nil && in_tex_uv != nil, "input tex is nil");
    OPS_CHECK(out_tex_y != nil && out_tex_uv != nil, "output tex is nil");

    id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
    OPS_CHECK(command_buffer != nil, "command_buffer is nil");
    id<MTLComputeCommandEncoder> command_encoder = [command_buffer computeCommandEncoder];
    OPS_CHECK(command_encoder != nil, "command_encoder is nil");

    if (in_width_nv12_ != in_tex_y.width || in_height_nv12_ != in_tex_y.height ) {
        in_width_nv12_ = in_tex_y.width;
        in_height_nv12_ = in_tex_y.height;
        in_width_ = in_width_nv12_;
        in_height_ = in_height_nv12_;

        OPS_CHECK(BMF_LITE_StsOk == MetalHelper::instance().gen_tex(&tex_ping_nv12_, MTLPixelFormatR8Unorm, in_width_nv12_, in_height_nv12_, MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite, MTLStorageModePrivate), 
                        "create tex0 error");
        OPS_CHECK(BMF_LITE_StsOk == MetalHelper::instance().gen_tex(&tex_pong_nv12_, MTLPixelFormatR8Unorm, in_width_nv12_, in_height_nv12_, MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite, MTLStorageModePrivate), 
                        "create tex1 error");
        first_run_ = true;
    }

    if (first_run_ == true) {
        [command_encoder setComputePipelineState:denoise_nv12_no_mix_];
        [command_encoder setTexture:in_tex_y   atIndex:0];
        [command_encoder setTexture:out_tex_y    atIndex:2];
        [command_encoder setTexture:tex_pong_nv12_    atIndex:3];
        [command_encoder setTexture:in_tex_uv    atIndex:4];
        [command_encoder setTexture:out_tex_uv    atIndex:5];
        [command_encoder dispatchThreads:MTLSizeMake((in_width_ + 1)/2, (in_height_ + 1)/2, 1) threadsPerThreadgroup:group_size_];
        [command_encoder endEncoding];
        [command_buffer commit];
        first_run_ = false;
    } else {
        [command_encoder setComputePipelineState:denoise_nv12_];
        [command_encoder setTexture:in_tex_y   atIndex:0];
        [command_encoder setTexture:tex_ping_nv12_   atIndex:1];
        [command_encoder setTexture:out_tex_y    atIndex:2];
        [command_encoder setTexture:tex_pong_nv12_    atIndex:3];
        [command_encoder setTexture:in_tex_uv    atIndex:4];
        [command_encoder setTexture:out_tex_uv    atIndex:5];
        [command_encoder dispatchThreads:MTLSizeMake((in_width_ + 1)/2, (in_height_ + 1)/2, 1) threadsPerThreadgroup:group_size_];
        [command_encoder endEncoding];
        [command_buffer commit];
    }
    // swap ping/pong
    id<MTLTexture> tmp = tex_ping_nv12_;
    tex_ping_nv12_ = tex_pong_nv12_;
    tex_pong_nv12_ = tmp;
   
    return BMF_LITE_StsOk;
}


} // namespace metal
} // namespace bmf_lite
