package com.bytedance.hmp;

public enum ColorPrimaries {
    CP_RESERVED0(0),
    CP_BT709(1),  ///< also ITU-R BT1361 / IEC 61966-2-4 / SMPTE RP177 Annex B
    CP_UNSPECIFIED(2),
    CP_RESERVED(3),
    CP_BT470M(4),  ///< also FCC Title 47 Code of Federal Regulations 73.682 (a)(20)

    CP_BT470BG(5),  ///< also ITU-R BT601-6 625 / ITU-R BT1358 625 / ITU-R BT1700 625 PAL & SECAM
    CP_SMPTE170M(6),  ///< also ITU-R BT601-6 525 / ITU-R BT1358 525 / ITU-R BT1700 NTSC
    CP_SMPTE240M(7),  ///< functionally identical to above
    CP_FILM(8),  ///< colour filters using Illuminant C
    CP_BT2020(9),  ///< ITU-R BT2020
    CP_SMPTE428(10), ///< SMPTE ST 428-1 (CIE 1931 XYZ)
    CP_SMPTEST428_1(10),
    CP_SMPTE431(11), ///< SMPTE ST 431-2 (2011) / DCI P3
    CP_SMPTE432(12), ///< SMPTE ST 432-1 (2010) / P3 D65 / Display P3
    CP_EBU3213(22), ///< EBU Tech. 3213-E / JEDEC P22 phosphors
    CP_JEDEC_P22(22),
    CP_NB(23);                ///< Not part of ABI

    private final int value;

    ColorPrimaries(final int v){ value = v; }

    public int getValue() { return value; }
}
