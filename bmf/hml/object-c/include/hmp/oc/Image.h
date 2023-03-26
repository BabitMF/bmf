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
#ifndef _HMP_OC_IMAGE_H
#define _HMP_OC_IMAGE_H

#import <Foundation/Foundation.h>
#import <AVFoundation/AVFoundation.h>
#import <hmp/oc/ScalarType.h>
#import <hmp/oc/Formats.h>
#import <hmp/oc/Tensor.h>


@interface HmpImage: NSObject{
}

- (instancetype) init : (HmpTensor*) data : (HmpChannelFormat) format;

- (instancetype) init : (int) width : (int) height : (int) nchannels : (HmpChannelFormat) format : (HmpScalarType) dtype;

- (instancetype) initFromPtr: (void*) ptr : (bool) own;

- (void*)ptr;

- (void) dealloc;

- (bool) defined;

- (NSString*) description;

- (HmpChannelFormat) format;

- (void) set_color_model: (HmpColorModel*) cm;

- (HmpColorModel*) color_model;

- (int) wdim;

- (int) hdim;

- (int) cdim;

- (int) width;

- (int) height;

- (int) nchannels;

- (HmpScalarType) dtype;

- (HmpDevice*) device;

- (HmpTensor*) data;

- (void*) unsafe_data;

- (HmpImage*) copy_: (HmpImage*) from;

- (HmpImage*) clone;

- (HmpImage*) crop : (int)left : (int)top : (int)width : (int) height;

- (HmpImage*) select : (int) channel; 


@end //HmpImage


@interface HmpFrame: NSObject{
@protected
    void *_impl;
    bool _own;
}

- (instancetype) init : (int) width : (int) height : (HmpPixelInfo*) pix_info;
- (instancetype) initFromData: (NSMutableArray*) data : (int) width : (int) height : (HmpPixelInfo*) pix_info;
// Frame must locked before use if it is init from pixelbuffer
- (instancetype) initFromPixelBuffer: (CVPixelBufferRef) pix_buffer;

- (instancetype) initFromPtr: (void*) ptr : (bool) own;

- (void*)ptr;

- (void) dealloc;

- (bool) defined;

- (NSString*) description;

- (HmpPixelInfo*) pix_info;
- (HmpPixelFormat) format;
- (int) width;
- (int) height;
- (HmpScalarType) dtype;
- (HmpDevice*) device;
- (int64_t) nplanes;
- (HmpTensor*) plane: (int) p;
- (void*) plane_data : (int) p;
- (HmpFrame*) copy_: (HmpFrame*) from;
- (HmpFrame*) crop : (int)left : (int)top : (int)width : (int) height;
// only support rgbx format for mobile build(HMP_ENABLE_MOBILE=ON)
- (HmpImage*) to_image : (HmpChannelFormat) cformat;
+ (HmpFrame*) from_image : (HmpImage*) image : (HmpPixelInfo*) pix_info;

@end //HmpFrame

#endif