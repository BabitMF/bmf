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
#include <map>
#include <hmp/core/logging.h>
#include <hmp/imgproc.h>
#include <hmp/imgproc/formats.h>
#include <hmp/format.h>

namespace hmp{


static ColorRange infer_color_range(PixelFormat fmt, ColorRange hint = CR_UNSPECIFIED)
{
    if(fmt == PF_YUVJ420P && hint == CR_UNSPECIFIED){
        return CR_JPEG;
    }
    else{
        return hint;
    }
}

ColorModel::ColorModel()
    : ColorModel(
        CS_UNSPECIFIED, CR_UNSPECIFIED,
        CP_UNSPECIFIED, CTC_UNSPECIFIED)
{
}


ColorModel::ColorModel(ColorSpace cs, ColorRange cr)
    : ColorModel(cs, cr,
                 CP_UNSPECIFIED, CTC_UNSPECIFIED)
{
}

ColorModel::ColorModel(ColorPrimaries cp, ColorTransferCharacteristic ctc)
    : ColorModel(CS_UNSPECIFIED, CR_UNSPECIFIED,
                 cp, ctc)
{
}

ColorModel::ColorModel(ColorSpace cs, ColorRange cr, ColorPrimaries cp, ColorTransferCharacteristic ctc)
{
    cm_ = cs | (cr<<8) | (cp << 16) | (ctc << 24); 
}

ColorSpace ColorModel::space() const
{
    return ColorSpace(cm_ & 0xff);
}

ColorRange ColorModel::range() const
{
    return ColorRange((cm_ >> 8) & 0xff);
}


ColorPrimaries ColorModel::primaries() const
{
    return ColorPrimaries((cm_ >> 16) & 0xff);
}


ColorTransferCharacteristic ColorModel::transfer_characteristic() const
{
    return ColorTransferCharacteristic((cm_ >> 24) & 0xff);
}


PixelInfo::PixelInfo()
    : format_(PF_NONE)
{
}

PixelInfo::PixelInfo(PixelFormat format, ColorModel color_model)
    : format_(format), color_model_(color_model)
{
}

PixelInfo::PixelInfo(PixelFormat format, ColorSpace cs, ColorRange cr)
    : format_(format), color_model_(cs, infer_color_range(format, cr))
{
}

PixelInfo::PixelInfo(PixelFormat format, ColorPrimaries cp, ColorTransferCharacteristic ctc)
    : format_(format), color_model_(CS_UNSPECIFIED, infer_color_range(format), cp, ctc)
{
}

bool PixelInfo::is_rgbx() const
{
    return PixelFormatDesc(format()).nplanes() == 1;
}


ColorSpace PixelInfo::infer_space() const
{
    if(space() != CS_UNSPECIFIED){
        return space();
    }
    else{
        if(format() == PF_NV12 || format() == PF_NV21){
            return CS_BT470BG;
        }
        else{
            return CS_BT709;
        }
    }
}


const static int sMaxPlanes = 4;


std::string stringfy(const PixelInfo &pix_info)
{
    return fmt::format("PixelInfo({}, {}, {}, {}, {})",
        pix_info.format(), pix_info.space(), pix_info.range(), 
        pix_info.primaries(), pix_info.transfer_characteristic());
}


std::string stringfy(const PixelFormat &format)
{
    switch (format)
    {
#define STRINGFY_CASE(name) case PixelFormat::name : return "k"#name;
    HMP_FORALL_PIXEL_FORMATS(STRINGFY_CASE)
#undef STRINGFY_CASE
    default:
        return fmt::format("PixelFormat({})", static_cast<int>(format));
    }
}


std::string stringfy(const ChannelFormat &format)
{
    switch (format)
    {
        case ChannelFormat::NCHW: return "kNCHW";
        case ChannelFormat::NHWC: return "kNHWC";
        default:
            return fmt::format("ChannelFormat({})", static_cast<int>(format));
    }
}

std::string stringfy(const ImageRotationMode &mode)
{
    switch (mode)
    {
        case ImageRotationMode::Rotate0: return "kRotate0";
        case ImageRotationMode::Rotate90: return "kRotate90";
        case ImageRotationMode::Rotate180: return "kRotate180";
        case ImageRotationMode::Rotate270: return "kRotate270";
        default:
            return fmt::format("ImageRotationMode({})", static_cast<int>(mode));
    };
}


std::string stringfy(const ImageFilterMode &mode)
{
    switch (mode)
    {
        case ImageFilterMode::Nearest : return "kNearest";
        case ImageFilterMode::Bilinear : return "kBilinear";
        case ImageFilterMode::Bicubic : return "kBicubic";
        default:
            return fmt::format("ImageFilterMode({})", static_cast<int>(mode));
    }
}


std::string stringfy(const ImageAxis &axis)
{
    switch(axis){
        case ImageAxis::Horizontal : return "kHorizontal";
        case ImageAxis::Vertical : return "kVertical";
        case ImageAxis::HorizontalAndVertical: return "kHorizontalAndVertical";
        default:
            return fmt::format("ImageAxis({})", static_cast<int>(axis));
    }
}


struct PixelFormatDesc::Private{
    PixelFormat format; //
    ScalarType dtype; //
    int nplanes; //
    int32_t ratio[sMaxPlanes]; //ratio related to first plane, packed like `(height_ratio << 8) | width_ratio<<4 | channels`
};

static PixelFormatDesc::Private sPixelFormatMetas[]{
    //
    PixelFormatDesc::Private{PixelFormat::PF_YUV420P, kUInt8, 3, {0x111, 0x221, 0x221, 0x0}},
    PixelFormatDesc::Private{PixelFormat::PF_YUVJ420P, kUInt8, 3, {0x111, 0x221, 0x221, 0x0}},
    PixelFormatDesc::Private{PixelFormat::PF_YUV422P, kUInt8, 3, {0x111, 0x121, 0x121, 0x0}},
    PixelFormatDesc::Private{PixelFormat::PF_YUV444P, kUInt8, 3, {0x111, 0x111, 0x111, 0x0}},
    PixelFormatDesc::Private{PixelFormat::PF_NV12,  kUInt8, 2, {0x111, 0x222, 0x00, 0x0}},
    PixelFormatDesc::Private{PixelFormat::PF_NV21,  kUInt8, 2, {0x111, 0x222, 0x00, 0x0}},
    PixelFormatDesc::Private{PixelFormat::PF_P010LE,  kUInt16, 2, {0x111, 0x222, 0x00, 0x0}}, //
    PixelFormatDesc::Private{PixelFormat::PF_P016LE,  kUInt16, 2, {0x111, 0x222, 0x00, 0x0}}, //
    PixelFormatDesc::Private{PixelFormat::PF_YUV422P10LE, kUInt16, 3, {0x111, 0x121, 0x121, 0x0}},

    PixelFormatDesc::Private{PixelFormat::PF_GRAY8,  kUInt8, 1, {0x111, 0x00, 0x00, 0x0}}, //
    PixelFormatDesc::Private{PixelFormat::PF_YUVA420P,  kUInt8, 4, {0x111, 0x221, 0x221, 0x111}}, //
    PixelFormatDesc::Private{PixelFormat::PF_GRAY16,  kUInt16, 1, {0x111, 0x00, 0x00, 0x0}}, //
    PixelFormatDesc::Private{PixelFormat::PF_RGB24,  kUInt8, 1, {0x113, 0x00, 0x00, 0x0}}, //
    PixelFormatDesc::Private{PixelFormat::PF_BGR24,  kUInt8, 1, {0x113, 0x00, 0x00, 0x0}}, //
    PixelFormatDesc::Private{PixelFormat::PF_RGB48,  kUInt16, 1, {0x113, 0x00, 0x00, 0x0}}, //
    PixelFormatDesc::Private{PixelFormat::PF_YA8,    kUInt8,  1, {0x112, 0x00, 0x00, 0x0}}, //
    PixelFormatDesc::Private{PixelFormat::PF_RGBA32,  kUInt8, 1, {0x114, 0x00, 0x00, 0x0}}, //
    PixelFormatDesc::Private{PixelFormat::PF_ARGB32,  kUInt8, 1, {0x114, 0x00, 0x00, 0x0}}, //
    PixelFormatDesc::Private{PixelFormat::PF_ABGR32,  kUInt8, 1, {0x114, 0x00, 0x00, 0x0}}, //
    PixelFormatDesc::Private{PixelFormat::PF_BGRA32,  kUInt8, 1, {0x114, 0x00, 0x00, 0x0}}, //
    PixelFormatDesc::Private{PixelFormat::PF_RGBA64,  kUInt16, 1, {0x114, 0x00, 0x00, 0x0}}, //
};

#define RC(v) ((v)&0xf)
#define RW(v) (((v)>>4)&0xf)
#define RH(v) (((v)>>8)&0xf)


PixelFormatDesc::PixelFormatDesc(int format)
    : pix_format_(format)
{
    for(size_t i = 0; i < sizeof(sPixelFormatMetas)/sizeof(Private); ++i){
        if(sPixelFormatMetas[i].format == format){
            meta_ = &sPixelFormatMetas[i];
        }
    }
}

int PixelFormatDesc::nplanes() const
{
    HMP_REQUIRE(defined(), "PixelFormat {} is not supported", pix_format_);
    return meta_->nplanes;
}

ScalarType PixelFormatDesc::dtype() const
{
    HMP_REQUIRE(defined(), "PixelFormat {} is not supported", pix_format_);
    return meta_->dtype;
}

int PixelFormatDesc::format() const
{
    return pix_format_;
}


int PixelFormatDesc::channels(int plane) const
{
    HMP_REQUIRE(defined(), "PixelFormat {} is not supported", pix_format_);
    HMP_REQUIRE(plane < meta_->nplanes, 
        "Plane index {} is out of range {}", plane, meta_->nplanes);
    return RC(meta_->ratio[plane]);
}

int PixelFormatDesc::infer_width(int width, int plane) const
{
    HMP_REQUIRE(defined(), "PixelFormat {} is not supported", pix_format_);
    HMP_REQUIRE(plane < meta_->nplanes, 
        "Plane index {} is out of range {}", plane, meta_->nplanes);
    return width/RW(meta_->ratio[plane]);
}

int PixelFormatDesc::infer_height(int height, int plane) const
{
    HMP_REQUIRE(defined(), "PixelFormat {} is not supported", pix_format_);
    HMP_REQUIRE(plane < meta_->nplanes, 
        "Plane index {} is out of range {}", plane, meta_->nplanes);
    return height/RH(meta_->ratio[plane]);
}


int PixelFormatDesc::infer_nitems(int width, int height) const
{
    HMP_REQUIRE(defined(), "PixelFormat {} is not supported", pix_format_);
    int nitems = 0;
    for(int i = 0; i < nplanes(); ++i){
        nitems += infer_nitems(width, height, i);
    }
    return nitems;
}

int PixelFormatDesc::infer_nitems(int width, int height, int plane) const
{
    HMP_REQUIRE(defined(), "PixelFormat {} is not supported", pix_format_);
    return infer_width(width, plane) * infer_height(height, plane) * channels(plane);
}


} //namespace hmp