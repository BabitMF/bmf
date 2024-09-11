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

#ifndef _BMFLITE_DEMO_MTKVIEW_RENDER_H_
#define _BMFLITE_DEMO_MTKVIEW_RENDER_H_

#import <AVFoundation/AVFoundation.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>

typedef enum BmfViewRenderMode {
    MTL_RGBA = 0,
    MTL_NV12 = 1,
    MTL_YUV420 = 2,
} BmfViewRenderMode;

@interface BmfLiteViewRender : NSObject <MTKViewDelegate>

/*
 * 0 not rotated
 * 1 rotate,left and right flip
 * 2 rotate
 */
- (nonnull instancetype)initWithMetalKitView:(nonnull MTKView *)mtkView
                               WhetherRotate:(int)rotated;

- (void)drawInMTKView:(nonnull MTKView *)view;

- (void)setMTLTexture:(id<MTLTexture>)
                 tex0:(__strong id<MTLTexture>)tex1
                     :(__strong id<MTLTexture>)tex2
                     :(id<MTLTexture>)c_tex0
                     :(__strong id<MTLTexture>)c_tex1
                     :(__strong id<MTLTexture>)c_tex2
                     :(CVPixelBufferRef)buf;

- (void)setRenderPipelineConfig:(OSType)
                         format:(int)frame_width
                               :(int)frame_height
                               :(int)view_width
                               :(int)view_height
                               :(bool)compare
                               :(float)line;

- (void)setSliderValue:(float)value;

- (void)dealloc;
@end

#endif /* _BMFLITE_DEMO_MTKVIEW_RENDER_H_ */
