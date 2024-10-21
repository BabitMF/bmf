package com.bytedance.hmp;

public enum ColorTransferCharacteristic {
    CTC_RESERVED0(0),
    CTC_BT709(1),  ///< also ITU-R BT1361
    CTC_UNSPECIFIED(2),
    CTC_RESERVED(3),
    CTC_GAMMA22(4),  ///< also ITU-R BT470M / ITU-R BT1700 625 PAL & SECAM
    CTC_GAMMA28(5),  ///< also ITU-R BT470BG
    CTC_SMPTE170M(6),  ///< also ITU-R BT601-6 525 or 625 / ITU-R BT1358 525 or 625 / ITU-R BT1700 NTSC
    CTC_SMPTE240M(7),
    CTC_LINEAR(8),  ///< "Linear transfer characteristics"
    CTC_LOG(9),  ///< "Logarithmic transfer characteristic (100:1 range)"
    CTC_LOG_SQRT(10), ///< "Logarithmic transfer characteristic (100 * Sqrt(10) : 1 range)"
    CTC_IEC61966_2_4(11), ///< IEC 61966-2-4
    CTC_BT1361_ECG(12), ///< ITU-R BT1361 Extended Colour Gamut
    CTC_IEC61966_2_1(13), ///< IEC 61966-2-1 (sRGB or sYCC)
    CTC_BT2020_10(14), ///< ITU-R BT2020 for 10-bit system
    CTC_BT2020_12(15), ///< ITU-R BT2020 for 12-bit system
    CTC_SMPTE2084(16), ///< SMPTE ST 2084 for 10-, 12-, 14- and 16-bit systems
    CTC_SMPTEST2084(16),
    CTC_SMPTE428(17), ///< SMPTE ST 428-1
    CTC_SMPTEST428_1(17),
    CTC_ARIB_STD_B67(18), ///< ARIB STD-B67, known as "Hybrid log-gamma"
    CTC_NB(19);                 ///< Not part of ABI

    private final int value;

    ColorTransferCharacteristic(final int v){ value = v; }

    public int getValue() { return value; }
}
