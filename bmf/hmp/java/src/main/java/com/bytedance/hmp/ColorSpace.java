package com.bytedance.hmp;

public enum ColorSpace {
    CS_RGB(0),  ///< order of coefficients is actually GBR, also IEC 61966-2-1 (sRGB)
    CS_BT709(1),  ///< also ITU-R BT1361 / IEC 61966-2-4 xvYCC709 / SMPTE RP177 Annex B
    CS_UNSPECIFIED(2),
    CS_RESERVED(3),
    CS_FCC(4),  ///< FCC Title 47 Code of Federal Regulations 73.682 (a)(20)
    CS_BT470BG(5),  ///< also ITU-R BT601-6 625 / ITU-R BT1358 625 / ITU-R BT1700 625 PAL & SECAM / IEC 61966-2-4 xvYCC601
    CS_SMPTE170M(6),  ///< also ITU-R BT601-6 525 / ITU-R BT1358 525 / ITU-R BT1700 NTSC
    CS_SMPTE240M(7),  ///< functionally identical to above
    CS_YCGCO(8),  ///< Used by Dirac / VC-2 and H.264 FRext, see ITU-T SG16
    CS_YCOCG(8),
    CS_BT2020_NCL(9),  ///< ITU-R BT2020 non-constant luminance system
    CS_BT2020_CL(10), ///< ITU-R BT2020 constant luminance system
    CS_SMPTE2085(11), ///< SMPTE 2085, Y'D'zD'x
    CS_CHROMA_DERIVED_NCL(2), ///< Chromaticity-derived non-constant luminance system
    CS_CHROMA_DERIVED_CL(13), ///< Chromaticity-derived constant luminance system
    CS_ICTCP(14), ///< ITU-R BT.2100-0, ICtCp
    CS_NB(15);

    private final int value;

    ColorSpace(final int v){ value = v; }

    public int getValue() { return value; }
}
