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

#ifndef _BMFLITE_DEMO_TOOLKIT_H_
#define _BMFLITE_DEMO_TOOLKIT_H_

#import "BmfLiteDemoMacro.h"
#import <simd/simd.h>
#import <AVFoundation/AVFoundation.h>

BMFLITE_DEMO_NAMESPACE_BEGIN

size_t getBitsCount(OSType pixelFmt);
size_t getBitsCount(MTLPixelFormat textureFmt);

bool isHDR(CVBufferRef pixelBuffer);

struct ColorConversion {
    matrix_float3x3 matrix;
    vector_float3 offset;
    // rangeMin/rangeMax is min and max value of yuv or rgb
    // currently used in RGBtoYUV to clamp the output yuv
    // it can be calculated from Scale(n)bitVideoRange and
    // Offset(n)bitVideoRange
    vector_float3 rangeMin;
    vector_float3 rangeMax;
    float line;
};

// bt2020 video-range 10bit
ColorConversion &getRGBtoYUVBt2020VideoRange10bit();
ColorConversion &getYUVtoRGBBt2020VideoRange10bit();

// bt709 video-range 8bit
ColorConversion &getRGBtoYUVBt709VideoRange8bit();
ColorConversion &getYUVtoRGBBt709VideoRange8bit();

BMFLITE_DEMO_NAMESPACE_END

#endif /* _BMFLITE_DEMO_TOOLKIT_H_ */
