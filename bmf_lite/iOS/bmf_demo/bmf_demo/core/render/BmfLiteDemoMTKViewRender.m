/*
 * Copyright 2024 Babit Authors
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

#import "BmfLiteDemoMTKViewRender.h"
#import "BmfLiteDemoMacro.h"
#import "BmfLiteDemoErrorCode.h"
#import "BmfLiteDemoLog.h"
#import "BmfLiteDemoToolKit.h"
#include <mutex>

USE_BMFLITE_DEMO_NAMESPACE

@implementation BmfLiteViewRender {
    id<MTLRenderPipelineState> render_pipeline_state_;
    id<MTLDevice> device_;
    id<MTLCommandQueue> command_queue_;
    MTKView* mtk_view_;
    vector_uint2 viewport_size_;
    BmfViewRenderMode render_mode_;
    id<MTLTexture> tex0_;
    id<MTLTexture> tex1_;
    id<MTLTexture> tex2_;
    id<MTLCommandBuffer> last_command_buffer_;
    id<MTLBuffer> curColorConversionBuffer;
    id<MTLBuffer> vertexBuffer_;
    id <MTLSamplerState> samplerState_;
    id<MTLLibrary> default_library_;
    int video_width_;
    int video_height_;
    int world_width_;
    int world_height_;
    float video_ratio_;
    float world_ratio_;
    int rotated_;
    // for compare
    bool compare_;
    CVPixelBufferRef pre_buf_;
    id<MTLTexture> c_tex0_;
    id<MTLTexture> c_tex1_;
    id<MTLTexture> c_tex2_;
    std::mutex mtx_;
}

- (nonnull instancetype)initWithMetalKitView:(nonnull MTKView *)mtkView WhetherRotate : (int) rotated {
    @autoreleasepool {
        self = [super init];
        if (nil == self) {
            BMFLITE_DEMO_LOGE("BmfLite", "call initWithMTKView failed, in super init.");
            return nil;
        }
        rotated_ = rotated;
        device_ = MTLCreateSystemDefaultDevice();
        if (nil == device_) {
            BMFLITE_DEMO_LOGE("BmfLite", "call MTLCreateSystemDefaultDevice failed.");
            return nil;
        }
        command_queue_ = [device_ newCommandQueue];
        if (nil == command_queue_) {
            BMFLITE_DEMO_LOGE("BmfLite", "create command queue failed.");
        }

        default_library_ = [device_ newDefaultLibrary];
        curColorConversionBuffer = [device_ newBufferWithLength:sizeof(ColorConversion) options:MTLResourceStorageModeShared];
        // vertexBuffer_ = [device_ newBufferWithBytes:vertex length: sizeof(vertex) options:0];
        vertexBuffer_ = [device_ newBufferWithLength:sizeof(float[30]) options:MTLResourceStorageModeShared];
        self->mtk_view_ = mtkView;
        self->mtk_view_.device = self->device_;
        pre_buf_ = nil;
        return self;
    }
}

- (void) setRenderPipelineConfig: (OSType) format : (int) frame_width : (int) frame_height : (int) view_width : (int) view_height :(bool)compare :(float)line
{
    std::lock_guard<std::mutex> lk(mtx_);
    video_ratio_ = (frame_width * 1.0) / frame_height;
    world_ratio_ = (view_width * 1.0) / view_height;
    MTLRenderPipelineDescriptor* render_pipeline_descriptor_ = [[MTLRenderPipelineDescriptor alloc] init];
    id<MTLFunction> vertex_function = [default_library_ newFunctionWithName:@"texture_vertex"];
    id<MTLFunction> fragment_function = nil;
    if (format == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange ||
        format == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange ||
        format == kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange) {
        if (compare) {
            if (!rotated_) {
                fragment_function = [default_library_ newFunctionWithName:@"nv12_rgb_compare"];
            } else {
                fragment_function = [default_library_ newFunctionWithName:@"nv12_rgb_compare_rotate"];
            }

        } else {
            fragment_function = [default_library_ newFunctionWithName:@"nv12_rgb"];
        }
        compare_ = compare;
    } else if (format != kCVPixelFormatType_420YpCbCr8Planar ||
               format != kCVPixelFormatType_420YpCbCr8PlanarFullRange) {
        fragment_function = [default_library_ newFunctionWithName:@"yuv_rgb"];
    } else {
        BMFLITE_DEMO_LOGE("BmfLite", "BmfRenderView not support format:%d", format);
    }
    render_pipeline_descriptor_.label = @"BmfViewRender Render Pipeline";
    render_pipeline_descriptor_.vertexFunction = vertex_function;
    render_pipeline_descriptor_.fragmentFunction = fragment_function;
    render_pipeline_descriptor_.colorAttachments[0].pixelFormat = mtk_view_.colorPixelFormat;
    NSError* err = nil;
    render_pipeline_state_ = [device_ newRenderPipelineStateWithDescriptor:render_pipeline_descriptor_ error:&err];
    if (nil == render_pipeline_state_) {
        BMFLITE_DEMO_LOGE(@"BmfLite", @"BmfRenderView create render pipeline state:%@", err);
    }
    
    ColorConversion color_conversion;
    if (kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange == format) {
        color_conversion = getYUVtoRGBBt2020VideoRange10bit();
    } else {
        color_conversion = getYUVtoRGBBt709VideoRange8bit();
    }
    color_conversion.line = line;
    memcpy(curColorConversionBuffer.contents, &color_conversion, sizeof(ColorConversion));

    MTLSamplerDescriptor *samplerDescriptor = [MTLSamplerDescriptor new];
    samplerDescriptor.minFilter = MTLSamplerMinMagFilterLinear;
    samplerDescriptor.magFilter = MTLSamplerMinMagFilterLinear;
   
    samplerState_ = [device_ newSamplerStateWithDescriptor:samplerDescriptor];
    if (rotated_ == 0) {
        float vertex[] = {
            -1.0,   -1.0,  0.0,   0.0,  1.0,
            1.0,   -1.0,  0.0,   1.0,  1.0,
            1.0,    1.0,  0.0,   1.0,  0.0,
            1.0,    1.0,  0.0,   1.0,  0.0,
            -1.0,   -1.0,  0.0,   0.0,  1.0,
            -1.0,    1.0,  0.0,   0.0,  0.0
        };
        memcpy(vertexBuffer_.contents, &vertex, sizeof(vertex));
    } else if (rotated_ == 1) {
        float vertex[] = {
            -1.0,   -1.0,  0.0,   1.0,  0.0,
             1.0,   -1.0,  0.0,   1.0,  1.0,
             1.0,    1.0,  0.0,   0.0,  1.0,
             1.0,    1.0,  0.0,   0.0,  1.0,
             -1.0,   -1.0,  0.0,   1.0,  0.0,
            -1.0,    1.0,  0.0,   0.0,  0.0
        };
        memcpy(vertexBuffer_.contents, &vertex, sizeof(vertex));
    } else if (rotated_ == 2) {
        float vertex[] = {
            -1.0,   -1.0,  0.0,   1.0,  1.0,
             1.0,   -1.0,  0.0,   1.0,  0.0,
             1.0,    1.0,  0.0,   0.0,  0.0,
             1.0,    1.0,  0.0,   0.0,  0.0,
             -1.0,   -1.0,  0.0,   1.0,  1.0,
            -1.0,    1.0,  0.0,   0.0,  1.0
        };
        memcpy(vertexBuffer_.contents, &vertex, sizeof(vertex));
    }
    vertexBuffer_.label = @"Bmf Vertices";
}

- (void)setSliderValue:(float)value {
    std::lock_guard<std::mutex> lk(mtx_);
//    char* ptr = (char *)curColorConversionBuffer.contents + sizeof(ColorConversion) - sizeof(float);
//    memcpy(ptr, &value, sizeof(float));
    ColorConversion color_conversion;
//    if (kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange == format) {
//        color_conversion = getYUVtoRGBBt2020VideoRange10bit();
//    } else {
        color_conversion = getYUVtoRGBBt709VideoRange8bit();
//    }
    if (rotated_ == 2) {
        color_conversion.line = 1 - value;
    } else {
        color_conversion.line = value;
    }
    memcpy(curColorConversionBuffer.contents, &color_conversion, sizeof(ColorConversion));
}

- (void) setMTLTexture:(id<MTLTexture>) tex0 : (__strong id<MTLTexture>) tex1 : (__strong id<MTLTexture>) tex2 :(id<MTLTexture>) c_tex0 : (__strong id<MTLTexture>) c_tex1 : (__strong id<MTLTexture>) c_tex2 : (CVPixelBufferRef) buf {
    std::lock_guard<std::mutex> lk(mtx_);
    self->tex0_ = tex0;
    self->tex1_ = tex1;
    self->tex2_ = tex0;
    self->c_tex0_ = c_tex0;
    self->c_tex1_ = c_tex1;
    self->c_tex2_ = c_tex0;
    if (pre_buf_ != nil) {
        CVPixelBufferRelease(pre_buf_);
        pre_buf_ = nil;
    }
    pre_buf_ = CVPixelBufferRetain(buf);
    
}

- (void) dealloc {
}

- (void)drawInMTKView:(nonnull MTKView *)view {
    std::lock_guard<std::mutex> lk(mtx_);
    if (nil == self->tex0_){
        return;
    }
    if (nil != last_command_buffer_) {
        MTLCommandBufferStatus status = last_command_buffer_.status;
        if (last_command_buffer_.status != MTLCommandBufferStatusCompleted && last_command_buffer_.status != MTLCommandBufferStatusError) {
            [last_command_buffer_ waitUntilCompleted];
        }
        last_command_buffer_ = nil;
        if (status == MTLCommandBufferStatusError) {
            BMFLITE_DEMO_LOGE("BmfLite", "command buffer exec failed.");
        }
    }
    MTLRenderPassDescriptor *render_pass_descriptor = view.currentRenderPassDescriptor;
    if (nil != render_pass_descriptor) {
        id<MTLCommandBuffer> command_buffer = [command_queue_ commandBuffer];
        if (nil == command_buffer) {
            BMFLITE_DEMO_LOGE("BmfLite", "create command buffer failed.");
            return;
        }
        command_buffer.label = @"BmfViewRender command buffer";
        id<MTLRenderCommandEncoder> command_encoder = [command_buffer renderCommandEncoderWithDescriptor:render_pass_descriptor];
        command_encoder.label = @"BmfViewRender command encoder";
        
        int video_h = tex0_.height;
        int video_w = tex0_.width;
        float real_h;
        if (rotated_ == 0) {
            real_h = video_h * (viewport_size_.x * 1.0f / video_w);
        } else {
            real_h = video_w * (viewport_size_.x * 1.0f / video_h);
        }
        float origin_y = ((viewport_size_.y - real_h) / 2);

        [command_encoder setViewport:(MTLViewport){0.0f, origin_y, (float)viewport_size_.x, real_h, -1.0f, 1.0f }];
        [command_encoder setRenderPipelineState:render_pipeline_state_];
        [command_encoder setVertexBuffer:vertexBuffer_ offset:0 atIndex:0 ];
        [command_encoder setFragmentSamplerState:samplerState_ atIndex:0];


        if (compare_) {
            if (rotated_ == 2) {
                [command_encoder setFragmentTexture:c_tex0_ atIndex:0];
                [command_encoder setFragmentTexture:c_tex1_ atIndex:1];
                [command_encoder setFragmentTexture:tex0_ atIndex:2];
                [command_encoder setFragmentTexture:tex1_ atIndex:3];
            } else {
                [command_encoder setFragmentTexture:tex0_ atIndex:0];
                [command_encoder setFragmentTexture:tex1_ atIndex:1];
                [command_encoder setFragmentTexture:c_tex0_ atIndex:2];
                [command_encoder setFragmentTexture:c_tex1_ atIndex:3];
            }
        } else {
            [command_encoder setFragmentTexture:tex0_ atIndex:0];
            [command_encoder setFragmentTexture:tex1_ atIndex:1];
        }
        [command_encoder setFragmentBuffer:curColorConversionBuffer offset:0 atIndex:0];
        [command_encoder drawPrimitives:(MTLPrimitiveTypeTriangle) vertexStart:0 vertexCount:6 instanceCount:1];
        [command_encoder endEncoding];
        [command_buffer presentDrawable:view.currentDrawable];
        [command_buffer commit];
        last_command_buffer_ = command_buffer;
    }
}

- (void)mtkView:(nonnull MTKView *)view drawableSizeWillChange:(CGSize)size {
    viewport_size_.x = size.width;
    viewport_size_.y = size.height;
}
@end
