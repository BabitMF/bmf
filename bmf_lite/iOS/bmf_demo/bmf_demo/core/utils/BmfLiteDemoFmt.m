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

#import "BmfLiteDemoErrorCode.h"
#import "BmfLiteDemoLog.h"
#import "BmfLiteDemoFmt.h"

@implementation BmfLiteDemoFmt {
    int32_t plane_ratio[3];
}

- (instancetype) initWithCVPixelBufferFormat : (OSType)fmt;
{
    self = [super init];
    if (nil == self) {
        return nil;
    }
    self.pixel_buffer_fmt = fmt;
    switch (self.pixel_buffer_fmt) {
        case kCVPixelFormatType_32BGRA:
            self.tex0_fmt = MTLPixelFormatBGRA8Unorm;
            self.tex1_fmt = MTLPixelFormatInvalid;
            self.tex2_fmt = MTLPixelFormatInvalid;
            self.plane_count = 1;
            self->plane_ratio[0] = 0x114;
            self->plane_ratio[1] = 0x00;
            self->plane_ratio[2] = 0x00;
            break;
        case kCVPixelFormatType_ARGB2101010LEPacked:
            self.tex0_fmt = MTLPixelFormatRGB10A2Unorm;
            self.tex1_fmt = MTLPixelFormatInvalid;
            self.tex2_fmt = MTLPixelFormatInvalid;
            self.plane_count = 1;
            self->plane_ratio[0] = 0x114;
            self->plane_ratio[1] = 0x00;
            self->plane_ratio[2] = 0x00;
            break;
        case kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange:
            self.tex0_fmt = MTLPixelFormatR8Unorm;
            self.tex1_fmt = MTLPixelFormatRG8Unorm;
            self.tex2_fmt = MTLPixelFormatInvalid;
            self.plane_count = 2;
            self->plane_ratio[0] = 0x111;
            self->plane_ratio[1] = 0x222;
            self->plane_ratio[2] = 0x00;
            break;
        case kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange:
            self.tex0_fmt = MTLPixelFormatR16Unorm;
            self.tex1_fmt = MTLPixelFormatRG16Unorm;
            self.tex2_fmt = MTLPixelFormatInvalid;
            self.plane_count = 2;
            self->plane_ratio[0] = 0x111;
            self->plane_ratio[1] = 0x222;
            self->plane_ratio[2] = 0x00;
            break;
        case kCVPixelFormatType_420YpCbCr8BiPlanarFullRange:
            self.tex0_fmt = MTLPixelFormatR8Unorm;
            self.tex1_fmt = MTLPixelFormatRG8Unorm;
            self.tex2_fmt = MTLPixelFormatInvalid;
            self.plane_count = 2;
            self->plane_ratio[0] = 0x111;
            self->plane_ratio[1] = 0x222;
            self->plane_ratio[2] = 0x00;
            break;
        case kCVPixelFormatType_420YpCbCr10BiPlanarFullRange:
            self.tex0_fmt = MTLPixelFormatR16Unorm;
            self.tex1_fmt = MTLPixelFormatRG16Unorm;
            self.tex2_fmt = MTLPixelFormatInvalid;
            self.plane_count = 2;
            self->plane_ratio[0] = 0x111;
            self->plane_ratio[1] = 0x222;
            self->plane_ratio[2] = 0x00;
            break;
        case kCVPixelFormatType_420YpCbCr8Planar:
            self.tex0_fmt = MTLPixelFormatR8Unorm;
            self.tex1_fmt = MTLPixelFormatR8Unorm;
            self.tex2_fmt = MTLPixelFormatR8Unorm;
            self.plane_count = 3;
            self->plane_ratio[0] = 0x111;
            self->plane_ratio[1] = 0x221;
            self->plane_ratio[2] = 0x221;
            break;
        case kCVPixelFormatType_420YpCbCr8PlanarFullRange:
            self.tex0_fmt = MTLPixelFormatR8Unorm;
            self.tex1_fmt = MTLPixelFormatR8Unorm;
            self.tex2_fmt = MTLPixelFormatR8Unorm;
            self.plane_count = 3;
            self->plane_ratio[0] = 0x111;
            self->plane_ratio[1] = 0x221;
            self->plane_ratio[2] = 0x221;
            break;
        case kCVPixelFormatType_OneComponent16Half:
            self.tex0_fmt = MTLPixelFormatR16Float;
            self.tex1_fmt = MTLPixelFormatInvalid;
            self.tex2_fmt = MTLPixelFormatInvalid;
            self.plane_count = 1;
            self->plane_ratio[0] = 0x114;
            self->plane_ratio[1] = 0x00;
            self->plane_ratio[2] = 0x00;
            break;
        default:
            BMFLITE_DEMO_LOGE("BmfLiteDemo", "not support cvpixelbuffer fmt:%d", self.pixel_buffer_fmt);
            self.tex0_fmt = MTLPixelFormatInvalid;
            self.tex1_fmt = MTLPixelFormatInvalid;
            self.tex2_fmt = MTLPixelFormatInvalid;
            self->plane_ratio[0] = 0x00;
            self->plane_ratio[1] = 0x00;
            self->plane_ratio[2] = 0x00;
            break;
    }
    return self;
}

- (MTLPixelFormat) getTexFormatByPlane : (int)plane;
{
    return (plane == 0) ? self.tex0_fmt : ((plane == 1) ? self.tex1_fmt : ((plane == 2) ? self.tex2_fmt : MTLPixelFormatInvalid));
}

- (int) getPlaneCount;
{
    return self.plane_count;
}

- (int) getWidthByPlaneIndex:(uint32_t) index WithOriginWidth : (int) width;
{
    return width / (((self->plane_ratio[index])>>4)&0xf);
}

- (int) getHeightByPlaneIndex:(int) index WithOriginHeight : (int) height;
{
    return height / (((self->plane_ratio[index])>>8)&0xf);
}

@end
