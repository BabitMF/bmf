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
#ifndef _HMP_OC_FORMATS_H
#define _HMP_OC_FORMATS_H

#import <Foundation/Foundation.h>

/** 
  * Map from AVColorPrimaries  
  * Chromaticity coordinates of the source primaries.
  * These values match the ones defined by ISO/IEC 23001-8_2013 § 7.1.
  */
typedef enum _HmpColorPrimaries{
    CP_RESERVED0   = 0,
    CP_BT709       = 1,  ///< also ITU-R BT1361 / IEC 61966-2-4 / SMPTE RP177 Annex B
    CP_UNSPECIFIED = 2,
    CP_RESERVED    = 3,
    CP_BT470M      = 4,  ///< also FCC Title 47 Code of Federal Regulations 73.682 (a)(20)

    CP_BT470BG     = 5,  ///< also ITU-R BT601-6 625 / ITU-R BT1358 625 / ITU-R BT1700 625 PAL & SECAM
    CP_SMPTE170M   = 6,  ///< also ITU-R BT601-6 525 / ITU-R BT1358 525 / ITU-R BT1700 NTSC
    CP_SMPTE240M   = 7,  ///< functionally identical to above
    CP_FILM        = 8,  ///< colour filters using Illuminant C
    CP_BT2020      = 9,  ///< ITU-R BT2020
    CP_SMPTE428    = 10, ///< SMPTE ST 428-1 (CIE 1931 XYZ)
    CP_SMPTEST428_1 = CP_SMPTE428,
    CP_SMPTE431    = 11, ///< SMPTE ST 431-2 (2011) / DCI P3
    CP_SMPTE432    = 12, ///< SMPTE ST 432-1 (2010) / P3 D65 / Display P3
    CP_EBU3213     = 22, ///< EBU Tech. 3213-E / JEDEC P22 phosphors
    CP_JEDEC_P22   = CP_EBU3213,
    CP_NB                ///< Not part of ABI
}HmpColorPrimaries;


/** 
 * map from AVColorTransferCharacteristic
 * Color Transfer Characteristic.
 * These values match the ones defined by ISO/IEC 23001-8_2013 § 7.2.
 */
typedef enum _HmpColorTransferCharacteristic{
    CTC_RESERVED0    = 0,
    CTC_BT709        = 1,  ///< also ITU-R BT1361
    CTC_UNSPECIFIED  = 2,
    CTC_RESERVED     = 3,
    CTC_GAMMA22      = 4,  ///< also ITU-R BT470M / ITU-R BT1700 625 PAL & SECAM
    CTC_GAMMA28      = 5,  ///< also ITU-R BT470BG
    CTC_SMPTE170M    = 6,  ///< also ITU-R BT601-6 525 or 625 / ITU-R BT1358 525 or 625 / ITU-R BT1700 NTSC
    CTC_SMPTE240M    = 7,
    CTC_LINEAR       = 8,  ///< "Linear transfer characteristics"
    CTC_LOG          = 9,  ///< "Logarithmic transfer characteristic (100:1 range)"
    CTC_LOG_SQRT     = 10, ///< "Logarithmic transfer characteristic (100 * Sqrt(10) : 1 range)"
    CTC_IEC61966_2_4 = 11, ///< IEC 61966-2-4
    CTC_BT1361_ECG   = 12, ///< ITU-R BT1361 Extended Colour Gamut
    CTC_IEC61966_2_1 = 13, ///< IEC 61966-2-1 (sRGB or sYCC)
    CTC_BT2020_10    = 14, ///< ITU-R BT2020 for 10-bit system
    CTC_BT2020_12    = 15, ///< ITU-R BT2020 for 12-bit system
    CTC_SMPTE2084    = 16, ///< SMPTE ST 2084 for 10-, 12-, 14- and 16-bit systems
    CTC_SMPTEST2084  = CTC_SMPTE2084,
    CTC_SMPTE428     = 17, ///< SMPTE ST 428-1
    CTC_SMPTEST428_1 = CTC_SMPTE428,
    CTC_ARIB_STD_B67 = 18, ///< ARIB STD-B67, known as "Hybrid log-gamma"
    CTC_NB                 ///< Not part of ABI
}HmpColorTransferCharacteristic;


/** 
 * Map from AVColorSpace
 * YUV colorspace type.
 * These values match the ones defined by ISO/IEC 23001-8_2013 § 7.3.
 */
typedef enum _HmpColorSpace{
    CS_RGB         = 0,  ///< order of coefficients is actually GBR, also IEC 61966-2-1 (sRGB)
    CS_BT709       = 1,  ///< also ITU-R BT1361 / IEC 61966-2-4 xvYCC709 / SMPTE RP177 Annex B
    CS_UNSPECIFIED = 2,
    CS_RESERVED    = 3,
    CS_FCC         = 4,  ///< FCC Title 47 Code of Federal Regulations 73.682 (a)(20)
    CS_BT470BG     = 5,  ///< also ITU-R BT601-6 625 / ITU-R BT1358 625 / ITU-R BT1700 625 PAL & SECAM / IEC 61966-2-4 xvYCC601
    CS_SMPTE170M   = 6,  ///< also ITU-R BT601-6 525 / ITU-R BT1358 525 / ITU-R BT1700 NTSC
    CS_SMPTE240M   = 7,  ///< functionally identical to above
    CS_YCGCO       = 8,  ///< Used by Dirac / VC-2 and H.264 FRext, see ITU-T SG16
    CS_YCOCG       = CS_YCGCO,
    CS_BT2020_NCL  = 9,  ///< ITU-R BT2020 non-constant luminance system
    CS_BT2020_CL   = 10, ///< ITU-R BT2020 constant luminance system
    CS_SMPTE2085   = 11, ///< SMPTE 2085, Y'D'zD'x
    CS_CHROMA_DERIVED_NCL = 12, ///< Chromaticity-derived non-constant luminance system
    CS_CHROMA_DERIVED_CL = 13, ///< Chromaticity-derived constant luminance system
    CS_ICTCP       = 14, ///< ITU-R BT.2100-0, ICtCp
    CS_NB
}HmpColorSpace;


/** 
 * map from AVColorRange
 * MPEG vs JPEG YUV range.
 */ 
typedef enum _HmpColorRange{
    CR_UNSPECIFIED = 0,
    CR_MPEG        = 1, ///< the normal 219*2^(n-8) "MPEG" YUV ranges
    CR_JPEG        = 2, ///< the normal     2^n-1   "JPEG" YUV ranges
    CR_NB               ///< Not part of ABI
}HmpColorRange;

// map from AVPixelFormat
typedef enum _HmpPixelFormat{
    PF_NONE         = -1,
    PF_YUV420P      = 0, ///< planar YUV 4:2:0, 12bpp, (1 Cr & Cb sample per 2x2 Y samples)
    PF_YUV422P      = 4, ///< planar YUV 4:2:2, 16bpp, (1 Cr & Cb sample per 2x1 Y samples)
    PF_YUV444P      = 5, ///< planar YUV 4:4:4, 24bpp, (1 Cr & Cb sample per 1x1 Y samples)
    PF_NV12         = 23,   ///< planar YUV 4:2:0, 12bpp, 1 plane for Y and 1 plane for the UV components, which are interleaved (first byte U and the following byte V)
    PF_NV21         = 24,   ///< as above, but U and V bytes are swapped

    PF_GRAY8        = 8,   ///<        Y        ,  8bpp
    PF_RGB24        = 2,   ///< packed RGB 8:8:8, 24bpp, RGBRGB...
    PF_BGR24        = 3,   ///< packed RGB 8:8:8, 24bpp, BGRBGR...

    PF_ARGB32       = 25,      ///< packed ARGB 8:8:8:8, 32bpp, ARGBARGB...
    PF_RGBA32       = 26,      ///< packed RGBA 8:8:8:8, 32bpp, RGBARGBA...
    PF_ABGR32       = 27,      ///< packed ABGR 8:8:8:8, 32bpp, ABGRABGR...
    PF_BGRA32       = 28,      ///< packed BGRA 8:8:8:8, 32bpp, BGRABGRA...

    PF_GRAY16       = 30, ///<        Y        , 16bpp, little-endian
    PF_YUVA420P     = 33, ///< planar YUV 4:2:0, 20bpp, (1 Cr & Cb sample per 2x2 Y & A samples)
    PF_RGB48        = 35, ///< packed RGB 16:16:16, 48bpp, 16R, 16G, 16B, the 2-byte value for each R/G/B component is stored as little-endian
    PF_YA8          = 58,       ///< 8 bits gray, 8 bits alpha
    PF_RGBA64       = 107, ///< packed RGBA 16:16:16:16, 64bpp, 16R, 16G, 16B, 16A, the 2-byte value for each R/G/B/A component is stored as little-endian

    PF_P010LE       = 161, ///< like NV12, with 10bpp per component, data in the high bits, zeros in the low bits, little-endian
    PF_P016LE       = 172, ///< like NV12, with 16bpp per component, little-endian
}HmpPixelFormat;


typedef enum _HmpChannelFormat {
    kNCHW = 0,
    kNHWC = 1
}HmpChannelFormat;


@interface HmpColorModel: NSObject{
@protected
    void *_impl;
    bool _own;
}

- (instancetype) init : (HmpColorSpace) cs : (HmpColorRange) cr;
- (instancetype) initEx : (HmpColorSpace) cs : (HmpColorRange) cr : (HmpColorPrimaries) cp : (HmpColorTransferCharacteristic) ctc;
- (instancetype) initFromPtr: (void*) ptr : (bool) own;

- (void*)ptr;

- (void) dealloc;

- (HmpColorRange) range;
- (HmpColorSpace) space;
- (HmpColorPrimaries) primaries;
- (HmpColorTransferCharacteristic) transfer_characteristic;

@end //HmpColorModel



@interface HmpPixelInfo: NSObject{
@protected
    void *_impl;
    bool _own;
}

- (instancetype) init : (HmpPixelFormat) format;
- (instancetype) initEx : (HmpPixelFormat) format : (HmpColorModel*) model;
- (instancetype) initFromPtr: (void*) ptr : (bool) own;

- (void*)ptr;

- (void) dealloc;

- (HmpPixelFormat) format;
- (HmpColorRange) range;
- (HmpColorSpace) space;
- (HmpColorPrimaries) primaries;
- (HmpColorTransferCharacteristic) transfer_characteristic;
- (HmpColorSpace) infer_space;
- (bool) is_rgbx;

@end //HmpPixelInfo

#endif