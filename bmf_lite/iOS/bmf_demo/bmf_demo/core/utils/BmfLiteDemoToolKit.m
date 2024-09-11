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

#import "BmfLiteDemoToolKit.h"

BMFLITE_DEMO_NAMESPACE_BEGIN

size_t getBitsCount(OSType pixelFmt) {
    switch (pixelFmt) {
        case kCVPixelFormatType_32BGRA:
        case kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange:
            return 8;
        case kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange:
        case kCVPixelFormatType_ARGB2101010LEPacked:
            return 10;
        default:
            return 8;
    }
}

size_t getBitsCount(MTLPixelFormat textureFmt) {
    switch (textureFmt) {
        case MTLPixelFormatRGB10A2Unorm:
            return 10;
        case MTLPixelFormatR16Unorm:
        case MTLPixelFormatRG16Unorm:
            return 16;
        default:
            return 8;
    }
}

bool isHDR(CVBufferRef pixelBuffer) {
    if (!pixelBuffer) {
        return false;
    }
    CFStringRef YCbCrMatrix = (CFStringRef)CVBufferGetAttachment(pixelBuffer, kCVImageBufferYCbCrMatrixKey, NULL);
    if (YCbCrMatrix != nil && CFStringCompare(YCbCrMatrix,kCVImageBufferYCbCrMatrix_ITU_R_2020,0) == kCFCompareEqualTo) {
        return true;
    }
    CFStringRef colorStr = (CFStringRef)CVBufferGetAttachment(pixelBuffer, kCVImageBufferColorPrimariesKey, NULL);
    if (colorStr != nil && CFStringCompare(colorStr,kCVImageBufferColorPrimaries_ITU_R_2020,0) == kCFCompareEqualTo) {
        return true;
    }
    CFStringRef transStr = (CFStringRef)CVBufferGetAttachment(pixelBuffer, kCVImageBufferTransferFunctionKey, NULL);
    if (@available(iOS 11.0, *)) {
        if (transStr != nil && (CFStringCompare(transStr,kCVImageBufferTransferFunction_SMPTE_ST_2084_PQ,0) == kCFCompareEqualTo ||
                                CFStringCompare(transStr,kCVImageBufferTransferFunction_ITU_R_2100_HLG,0) == kCFCompareEqualTo)) {
            return true;
        }
    } else {
        // Fallback on earlier versions
    }
    return false;
}

/**
 use this function to generate RGBtoYUV matrix
 include bt2020/bt709 , video-range/full-range , 10bit/8bit
 */
matrix_float3x3 getRGBtoYUVMatrix(float kr, float kg, float kb, vector_float3 scale) {
    return {(vector_float3){kr * scale[0], -0.5f*kr/(1.f-kb) * scale[1], 0.5f * scale[2]             },
            (vector_float3){kg * scale[0], -0.5f*kg/(1.f-kb) * scale[1], -0.5f*kg/(1.f-kr) * scale[2]},
            (vector_float3){kb * scale[0], 0.5f * scale[1],              -0.5f*kb/(1.f-kr) * scale[2]}};
}

/**
 use this function to generate YUVtoRGB matrix
 include bt2020/bt709 , video-range/full-range , 10bit/8bit
 */
matrix_float3x3 getYUVtoRGBMatrix(float kr, float kg, float kb, vector_float3 scale) {
    return {(vector_float3){1.f / scale[0],      1.f / scale[0],             1.f / scale[0]},
            (vector_float3){0.0f,                -kb/kg*(2-2*kb) / scale[1], (2-2*kb) / scale[1]},
            (vector_float3){(2-2*kr) / scale[2], -kr/kg*(2-2*kr) / scale[2], 0.0f}};
}

//kr kb kb,used to calculate color matrix
const float krBt2020 = 0.2627, kgBt2020 = 0.678, kbBt2020 = 0.0593;//bt2020
const float krBt709 = 0.2126, kgBt709 = 0.7152, kbBt709 = 0.0722;//bt709

//10bit video range
const vector_float3 Scale10bitVideoRange = {876.0/1023.0, 896.0/1023.0, 896.0/1023.0};
//const vector_float3 Offset10bitVideoRange = {64.0/1023.0, 0.5, 0.5};
const vector_float3 Offset10bitVideoRange = {30.0/1023.0, 0.5, 0.5};

//8bit video range
const vector_float3 Scale8bitVideoRange = {219.0/255.0, 224.0/255.0, 224.0/255.0};
const vector_float3 Offset8bitBVideoRange = {16.0/255.0, 0.5, 0.5};

//bt2020 video-range 10bit
ColorConversion& getRGBtoYUVBt2020VideoRange10bit() {
    static ColorConversion colorConversion = {
        .matrix = {
            getRGBtoYUVMatrix(krBt2020, kgBt2020, kbBt2020, Scale10bitVideoRange)
        },
        .offset = Offset10bitVideoRange,
        .rangeMin = {64.0/1023.0,64.0/1023.0,64.0/1023.0},
        .rangeMax = {940.0/1023.0,960.0/1023.0,960.0/1023.0},
    };
    return colorConversion;
}

ColorConversion& getYUVtoRGBBt2020VideoRange10bit() {
    static ColorConversion colorConversion = {
        .matrix = {
            getYUVtoRGBMatrix(krBt2020, kgBt2020, kbBt2020, Scale10bitVideoRange)
        },
        .offset = Offset10bitVideoRange,
        .rangeMin = {0,0,0},
        .rangeMax = {1,1,1},
    };
    return colorConversion;
}

//bt709 video-range 8bit
ColorConversion& getRGBtoYUVBt709VideoRange8bit() {
    static ColorConversion colorConversion = {
        .matrix = {
            getRGBtoYUVMatrix(krBt709, kgBt709, kbBt709, Scale8bitVideoRange)
        },
        .offset = Offset8bitBVideoRange,
        .rangeMin = {16.0/255.0,16.0/255.0,16.0/255.0},
        .rangeMax = {235.0/255.0,240/255.0,240/255.0},
    };
    return colorConversion;
}

ColorConversion& getYUVtoRGBBt709VideoRange8bit() {
    static ColorConversion colorConversion = {
        .matrix = {
            getYUVtoRGBMatrix(krBt709, kgBt709, kbBt709, Scale8bitVideoRange)
        },
        .offset = Offset8bitBVideoRange,
        .rangeMin = {0,0,0},
        .rangeMax = {1,1,1},
    };
    return colorConversion;
}

BMFLITE_DEMO_NAMESPACE_END
