package com.bytedance.hmp;

public enum PixelFormat {
    PF_NONE(-1),
    PF_YUV420P(0), ///< planar YUV 4:2:0, 12bpp, (1 Cr & Cb sample per 2x2 Y samples)
    PF_YUV422P(4), ///< planar YUV 4:2:2, 16bpp, (1 Cr & Cb sample per 2x1 Y samples)
    PF_YUV444P(5), ///< planar YUV 4:4:4, 24bpp, (1 Cr & Cb sample per 1x1 Y samples)
    PF_NV12(23),   ///< planar YUV 4:2:0, 12bpp, 1 plane for Y and 1 plane for the UV components, which are interleaved (first byte U and the following byte V)
    PF_NV21(24),   ///< as above, but U and V bytes are swapped

    PF_GRAY8(8),   ///<        Y        ,  8bpp
    PF_RGB24(2),   ///< packed RGB 8:8:8, 24bpp, RGBRGB...
    PF_RGBA32(26),   ///< packed ABGR 8:8:8:8, 32bpp, ABGRABGR...

    PF_GRAY16(30), ///<        Y        , 16bpp, little-endian
    PF_YUVA420P(33), ///< planar YUV 4:2:0, 20bpp, (1 Cr & Cb sample per 2x2 Y & A samples)
    PF_RGB48(35), ///< packed RGB 16:16:16, 48bpp, 16R, 16G, 16B, the 2-byte value for each R/G/B component is stored as little-endian
    PF_RGBA64(107); ///< packed RGBA 16:16:16:16, 64bpp, 16R, 16G, 16B, 16A, the 2-byte value for each R/G/B/A component is stored as little-endian


    private final int value;

    PixelFormat(final int v){ value = v; }

    public int getValue() { return value; }
}
