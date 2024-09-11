/*
 * Copyright 2019 bloc97
 *
 * This file is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This file is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 */
#include "metal/sr/sr.h"
#include "metal/metal_helper.h"
#include <numeric>

namespace bmf_lite {
namespace metal {

int Sr::init() {
    inited_ = false;

    OPS_CHECK(MetalHelper::instance().support_non_uniform_tg(), "not support non_uniform_tg");

    OPS_CHECK(BMF_LITE_StsOk == MetalHelper::instance().new_compute_pileline_state(&cps0_, @"pass0"), "can not find 0 function");
    OPS_CHECK(BMF_LITE_StsOk == MetalHelper::instance().new_compute_pileline_state(&cps1_, @"pass1"), "can not find 1 function");
    OPS_CHECK(BMF_LITE_StsOk == MetalHelper::instance().new_compute_pileline_state(&cps2_, @"pass2"), "can not find 2 function");
    OPS_CHECK(BMF_LITE_StsOk == MetalHelper::instance().new_compute_pileline_state(&cps3_, @"pass3"), "can not find 3 function");
    OPS_CHECK(BMF_LITE_StsOk == MetalHelper::instance().new_compute_pileline_state(&cps4_, @"pass4"), "can not find 4 function");
    OPS_CHECK(BMF_LITE_StsOk == MetalHelper::instance().new_compute_pileline_state(&rgba_to_nv12_, @"rgba_to_nv12"), "can not find 4 function");
    OPS_CHECK(BMF_LITE_StsOk == MetalHelper::instance().new_compute_pileline_state(&nv12_to_rgba_, @"nv12_to_rgba"), "can not find 4 function");
    group_size_ = MTLSizeMake(16, 16, 1);

    inited_ = true;
    return BMF_LITE_StsOk;
}

int Sr::run(id<MTLTexture> in_tex, id<MTLTexture> out_tex, id<MTLCommandQueue> command_queue) {
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

        OPS_CHECK(BMF_LITE_StsOk == MetalHelper::instance().gen_tex(&tex0_, MTLPixelFormatRGBA16Float, in_width_, in_height_, MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite, MTLStorageModePrivate), 
                        "create tex0 error");
        OPS_CHECK(BMF_LITE_StsOk == MetalHelper::instance().gen_tex(&tex1_, MTLPixelFormatRGBA16Float, in_width_, in_height_, MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite, MTLStorageModePrivate), 
                        "create tex1 error");
    }
    // 0
    [command_encoder setComputePipelineState:cps0_];
    [command_encoder setTexture:in_tex   atIndex:0];
    [command_encoder setTexture:tex0_    atIndex:1];
    [command_encoder dispatchThreads:MTLSizeMake(in_width_, in_height_, 1) threadsPerThreadgroup:group_size_];
    // 1
    [command_encoder setComputePipelineState:cps1_];
    [command_encoder setTexture:tex0_   atIndex:0];
    [command_encoder setTexture:tex1_    atIndex:1];
    [command_encoder dispatchThreads:MTLSizeMake(in_width_, in_height_, 1) threadsPerThreadgroup:group_size_];
    // 2
    [command_encoder setComputePipelineState:cps2_];
    [command_encoder setTexture:tex1_   atIndex:0];
    [command_encoder setTexture:tex0_    atIndex:1];
    [command_encoder dispatchThreads:MTLSizeMake(in_width_, in_height_, 1) threadsPerThreadgroup:group_size_];
    // 3
    [command_encoder setComputePipelineState:cps3_];
    [command_encoder setTexture:tex0_   atIndex:0];
    [command_encoder setTexture:tex1_    atIndex:1];
    [command_encoder dispatchThreads:MTLSizeMake(in_width_, in_height_, 1) threadsPerThreadgroup:group_size_];
    // 4
    [command_encoder setComputePipelineState:cps4_];
    [command_encoder setTexture:in_tex   atIndex:0];
    [command_encoder setTexture:tex1_    atIndex:1];
    [command_encoder setTexture:out_tex    atIndex:2];
    [command_encoder dispatchThreads:MTLSizeMake(out_tex.width, out_tex.height, 1) threadsPerThreadgroup:group_size_];    
    
    // commit
    [command_encoder endEncoding];
    [command_buffer commit];
   
    return BMF_LITE_StsOk;
}

int Sr::run(id<MTLTexture> in_tex_y, id<MTLTexture> in_tex_uv, id<MTLTexture> out_tex_y, id<MTLTexture> out_tex_uv, id<MTLCommandQueue> command_queue) {
    OPS_CHECK(inited_, "please call init first");
    OPS_CHECK(in_tex_y != nil, "input tex_y is nil");
    OPS_CHECK(in_tex_uv != nil, "input tex_uv is nil");
    OPS_CHECK(out_tex_y != nil, "output tex_y is nil");
    OPS_CHECK(out_tex_uv != nil, "output tex_uv is nil");

    id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
    OPS_CHECK(command_buffer != nil, "command_buffer is nil");
    id<MTLComputeCommandEncoder> command_encoder = [command_buffer computeCommandEncoder];
    OPS_CHECK(command_encoder != nil, "command_encoder is nil");

    if (in_width_nv12_ != in_tex_y.width || in_height_nv12_ != in_tex_y.height) {
        in_width_nv12_ = in_tex_y.width;
        in_height_nv12_ = in_tex_y.height;
        in_width_ = in_width_nv12_;
        in_height_ = in_height_nv12_;

        OPS_CHECK(BMF_LITE_StsOk == MetalHelper::instance().gen_tex(&tex0_, MTLPixelFormatRGBA16Float, in_width_nv12_, in_height_nv12_, MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite, MTLStorageModePrivate),
                        "create tex0 error");
        OPS_CHECK(BMF_LITE_StsOk == MetalHelper::instance().gen_tex(&tex1_, MTLPixelFormatRGBA16Float, in_width_nv12_, in_height_nv12_, MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite, MTLStorageModePrivate),
                        "create tex1 error");
        OPS_CHECK(BMF_LITE_StsOk == MetalHelper::instance().gen_tex(&tex_rgba_in_, MTLPixelFormatRGBA8Unorm, in_width_nv12_, in_height_nv12_, MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite, MTLStorageModePrivate),
                        "create tex_rgba_in error");
        OPS_CHECK(BMF_LITE_StsOk == MetalHelper::instance().gen_tex(&tex_rgba_out_, MTLPixelFormatRGBA8Unorm, out_tex_y.width, out_tex_y.height, MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite, MTLStorageModePrivate),
                        "create tex_rgba_out error");
    }
    // nv12 to rgba
    [command_encoder setComputePipelineState:nv12_to_rgba_];
    [command_encoder setTexture:tex_rgba_in_   atIndex:0];
    [command_encoder setTexture:in_tex_y    atIndex:1];
    [command_encoder setTexture:in_tex_uv    atIndex:2];
    [command_encoder dispatchThreads:MTLSizeMake((in_width_ + 1)/2, in_height_, 1) threadsPerThreadgroup:group_size_];
    // 0
    [command_encoder setComputePipelineState:cps0_];
    [command_encoder setTexture:tex_rgba_in_   atIndex:0];
    [command_encoder setTexture:tex0_    atIndex:1];
    [command_encoder dispatchThreads:MTLSizeMake(in_width_, in_height_, 1) threadsPerThreadgroup:group_size_];
    // 1
    [command_encoder setComputePipelineState:cps1_];
    [command_encoder setTexture:tex0_   atIndex:0];
    [command_encoder setTexture:tex1_    atIndex:1];
    [command_encoder dispatchThreads:MTLSizeMake(in_width_, in_height_, 1) threadsPerThreadgroup:group_size_];
    // 2
    [command_encoder setComputePipelineState:cps2_];
    [command_encoder setTexture:tex1_   atIndex:0];
    [command_encoder setTexture:tex0_    atIndex:1];
    [command_encoder dispatchThreads:MTLSizeMake(in_width_, in_height_, 1) threadsPerThreadgroup:group_size_];
    // 3
    [command_encoder setComputePipelineState:cps3_];
    [command_encoder setTexture:tex0_   atIndex:0];
    [command_encoder setTexture:tex1_    atIndex:1];
    [command_encoder dispatchThreads:MTLSizeMake(in_width_, in_height_, 1) threadsPerThreadgroup:group_size_];
    // 4
    [command_encoder setComputePipelineState:cps4_];
    [command_encoder setTexture:tex_rgba_in_   atIndex:0];
    [command_encoder setTexture:tex1_    atIndex:1];
    [command_encoder setTexture:tex_rgba_out_    atIndex:2];
    [command_encoder dispatchThreads:MTLSizeMake(tex_rgba_out_.width, tex_rgba_out_.height, 1) threadsPerThreadgroup:group_size_];
    //rgba to nv12
    [command_encoder setComputePipelineState:rgba_to_nv12_];
    [command_encoder setTexture:tex_rgba_out_   atIndex:0];
    [command_encoder setTexture:out_tex_y    atIndex:1];
    [command_encoder setTexture:out_tex_uv    atIndex:2];
    [command_encoder dispatchThreads:MTLSizeMake((out_tex_y.width + 1)/2, (out_tex_y.height + 1)/2, 1) threadsPerThreadgroup:group_size_];
    // commit
    [command_encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    return BMF_LITE_StsOk;
}


} // namespace metal
} // namespace bmf_lite
