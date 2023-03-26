package com.bytedance.hmp;

public enum ColorRange {
    CR_UNSPECIFIED(0),
    CR_MPEG(1), ///< the normal 219*2^(n-8) "MPEG" YUV ranges
    CR_JPEG(2), ///< the normal     2^n-1   "JPEG" YUV ranges
    CR_NB(3);               ///< Not part of ABI
	
    private final int value;

    ColorRange(final int v){ value = v; }

    public int getValue() { return value; }
}
