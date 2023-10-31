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
#pragma once

#include <stdint.h>
#include <hmp/core/scalar_type.h>

namespace hmp {

enum ColorPrimaries {
    CP_RESERVED0 = 0,
    CP_BT709 = 1,
    CP_UNSPECIFIED = 2,
    CP_RESERVED = 3,
    CP_BT470M = 4,

    CP_BT470BG = 5,
    CP_SMPTE170M = 6,
    CP_SMPTE240M = 7,
    CP_FILM = 8,
    CP_BT2020 = 9,
    CP_SMPTE428 = 10,
    CP_SMPTEST428_1 = CP_SMPTE428,
    CP_SMPTE431 = 11,
    CP_SMPTE432 = 12,
    CP_EBU3213 = 22,
    CP_JEDEC_P22 = CP_EBU3213,
    CP_NB
};

enum ColorTransferCharacteristic {
    CTC_RESERVED0 = 0,
    CTC_BT709 = 1,
    CTC_UNSPECIFIED = 2,
    CTC_RESERVED = 3,
    CTC_GAMMA22 = 4,
    CTC_GAMMA28 = 5,
    CTC_SMPTE170M = 6,
    CTC_SMPTE240M = 7,
    CTC_LINEAR = 8,
    CTC_LOG = 9,
    CTC_LOG_SQRT = 10,
    CTC_IEC61966_2_4 = 11,
    CTC_BT1361_ECG = 12,
    CTC_IEC61966_2_1 = 13,
    CTC_BT2020_10 = 14,
    CTC_BT2020_12 = 15,
    CTC_SMPTE2084 = 16,
    CTC_SMPTEST2084 = CTC_SMPTE2084,
    CTC_SMPTE428 = 17,
    CTC_SMPTEST428_1 = CTC_SMPTE428,
    CTC_ARIB_STD_B67 = 18,
    CTC_NB
};

enum ColorSpace {
    CS_RGB = 0,
    CS_BT709 = 1,
    CS_UNSPECIFIED = 2,
    CS_RESERVED = 3,
    CS_FCC = 4,
    CS_BT470BG = 5,
    CS_SMPTE170M = 6,
    CS_SMPTE240M = 7,
    CS_YCGCO = 8,
    CS_YCOCG = CS_YCGCO,
    CS_BT2020_NCL = 9,
    CS_BT2020_CL = 10,
    CS_SMPTE2085 = 11,
    CS_CHROMA_DERIVED_NCL = 12,
    CS_CHROMA_DERIVED_CL = 13,
    CS_ICTCP = 14,
    CS_NB
};

enum ColorRange { CR_UNSPECIFIED = 0, CR_MPEG = 1, CR_JPEG = 2, CR_NB };

enum PixelFormat {
    PF_NONE = -1,
    PF_YUV420P = 0,
    PF_YUV422P = 4,
    PF_YUV444P = 5,
    PF_NV12 = 23,
    PF_NV21 = 24,

    PF_GRAY8 = 8,
    PF_RGB24 = 2,
    PF_BGR24 = 3,

    PF_YUVJ420P = 12,

    PF_ARGB32 = 25,
    PF_RGBA32 = 26,
    PF_ABGR32 = 27,
    PF_BGRA32 = 28,

    PF_GRAY16 = 30,
    PF_YUVA420P = 33,
    PF_RGB48 = 35,
    PF_YA8 = 58,
    PF_BGR48 = 61,
    PF_RGBA64 = 107,

    PF_P010LE = 161,
    PF_P016LE = 172,
    PF_YUV422P10LE = 66,
    PF_YUV420P10LE = 64,
    PF_YUV444P10LE = 68,
};

#define HMP_FORALL_PIXEL_FORMATS(_)                                            \
    _(PF_YUV420P)                                                              \
    _(PF_YUV422P)                                                              \
    _(PF_YUV444P)                                                              \
    _(PF_NV12)                                                                 \
    _(PF_NV21)                                                                 \
    _(PF_GRAY8)                                                                \
    _(PF_RGB24)                                                                \
    _(PF_BGR24)                                                                \
    _(PF_RGBA32)                                                               \
    _(PF_BGRA32)                                                               \
    _(PF_GRAY16)                                                               \
    _(PF_YUVA420P)                                                             \
    _(PF_RGB48)                                                                \
    _(PF_YA8)                                                                  \
    _(PF_RGBA64)                                                               \
    _(PF_P010LE)                                                               \
    _(PF_P016LE)                                                               \
    _(PF_YUV422P10LE)                                                          \
    _(PF_YUV420P10LE)

HMP_API std::string stringfy(const PixelFormat &format);
PixelFormat HMP_API get_pixel_format(std::string pixfmt);

class HMP_API ColorModel {
  public:
    ColorModel();
    ColorModel(ColorSpace cs, ColorRange cr, ColorPrimaries cp,
               ColorTransferCharacteristic ctc);
    ColorModel(ColorSpace cs, ColorRange cr = CR_UNSPECIFIED);
    ColorModel(ColorPrimaries cp,
               ColorTransferCharacteristic ctc = CTC_UNSPECIFIED);

    ColorSpace space() const;
    ColorRange range() const;
    ColorPrimaries primaries() const;
    ColorTransferCharacteristic transfer_characteristic() const;

  private:
    uint32_t cm_;
};

class HMP_API PixelInfo {
  public:
    PixelInfo();
    PixelInfo(PixelFormat format, ColorModel color_model = {}, int align = 16);
    PixelInfo(PixelFormat format, ColorSpace cs, ColorRange cr = CR_UNSPECIFIED,
              int align = 16);
    PixelInfo(PixelFormat format, ColorPrimaries cp,
              ColorTransferCharacteristic ctc = CTC_UNSPECIFIED,
              int align = 16);

    PixelFormat format() const { return format_; }
    ColorSpace space() const { return color_model_.space(); }
    ColorRange range() const { return color_model_.range(); }
    ColorPrimaries primaries() const { return color_model_.primaries(); }
    ColorTransferCharacteristic transfer_characteristic() const {
        return color_model_.transfer_characteristic();
    }

    /**
     * @brief in case not specified
     *
     * @return ColorSpace
     */
    ColorSpace infer_space() const;

    const ColorModel &color_model() const { return color_model_; }

    bool is_rgbx() const;

    const int &alignment() const { return align_; }

  private:
    PixelFormat format_;
    ColorModel color_model_;
    int align_ = 16;
};

HMP_API std::string stringfy(const PixelInfo &pix_info);

enum class ChannelFormat : uint8_t { NCHW, NHWC };

enum CHW {
    CHW_C = 0,
    CHW_H = 1,
    CHW_W = 2,
};

enum HWC {
    HWC_H = 0,
    HWC_W = 1,
    HWC_C = 2,
};

enum NHWC {
    NHWC_N = 0,
    NHWC_H = 1,
    NHWC_W = 2,
    NHWC_C = 3,
};

enum NCHW {
    NCHW_N = 0,
    NCHW_C = 1,
    NCHW_H = 2,
    NCHW_W = 3,
};

const static ChannelFormat kNCHW = ChannelFormat::NCHW;
const static ChannelFormat kNHWC = ChannelFormat::NHWC;

HMP_API std::string stringfy(const ChannelFormat &format);

enum class ImageRotationMode : uint8_t {
    Rotate0,
    Rotate90,
    Rotate180,
    Rotate270,
};
const static ImageRotationMode kRotate0 = ImageRotationMode::Rotate0;
const static ImageRotationMode kRotate90 = ImageRotationMode::Rotate90;
const static ImageRotationMode kRotate180 = ImageRotationMode::Rotate180;
const static ImageRotationMode kRotate270 = ImageRotationMode::Rotate270;

HMP_API std::string stringfy(const ImageRotationMode &mode);

enum class ImageFilterMode : uint8_t { Nearest, Bilinear, Bicubic };
const static ImageFilterMode kNearest = ImageFilterMode::Nearest;
const static ImageFilterMode kBilinear = ImageFilterMode::Bilinear;
const static ImageFilterMode kBicubic = ImageFilterMode::Bicubic;

HMP_API std::string stringfy(const ImageFilterMode &mode);

enum class ImageAxis : uint8_t {
    Horizontal = 0x1,
    Vertical = 0x2,

    HorizontalAndVertical = 0x3
};
const static ImageAxis kHorizontal = ImageAxis::Horizontal;
const static ImageAxis kVertical = ImageAxis::Vertical;
const static ImageAxis kHorizontalAndVertical =
    ImageAxis::HorizontalAndVertical;

HMP_API std::string stringfy(const ImageAxis &axis);

class HMP_API PixelFormatDesc {
  public:
    struct Private;

    PixelFormatDesc() {}
    PixelFormatDesc(int format);

    //
    bool defined() const { return meta_ != nullptr; }

    int nplanes() const;
    ScalarType dtype() const;
    int format() const;
    int channels(int plane = 0) const;

    int infer_width(int width, int plane = 0) const;
    int infer_height(int height, int plane = 0) const;
    int infer_nitems(int width, int height) const;
    int infer_nitems(int width, int height, int plane) const;

  private:
    int pix_format_ = PF_NONE;
    const Private *meta_ = nullptr;
};

enum class RGBFormat : uint8_t  {
    RGB,
    BGR,
};

const static RGBFormat kRGB = RGBFormat::RGB;
const static RGBFormat kBGR = RGBFormat::BGR;

} // namespace hmp
