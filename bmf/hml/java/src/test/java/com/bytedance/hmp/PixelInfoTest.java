package com.bytedance.hmp;

import junit.framework.TestCase;

public class PixelInfoTest extends TestCase {
    public void testPixelInfoCreate()
    {
        //
        PixelInfo pi0 = new PixelInfo(PixelFormat.PF_YUV420P);
        assertTrue(pi0.format() == PixelFormat.PF_YUV420P);
        assertTrue(pi0.space() == ColorSpace.CS_UNSPECIFIED);
        assertTrue(pi0.primaries() == ColorPrimaries.CP_UNSPECIFIED);
        assertTrue(pi0.range() == ColorRange.CR_UNSPECIFIED);
        assertTrue(pi0.colorTransferCharacteristic() == ColorTransferCharacteristic.CTC_UNSPECIFIED);
        assertTrue(pi0.inferSpace() == ColorSpace.CS_BT709);
        assertTrue(pi0.own);
        ColorModel cm0 = pi0.colorModel();
        assertFalse(cm0.own);
        assertFalse(pi0.isRgbx());
        assertTrue(pi0.toString().length() > 0);
        pi0.free();

        //
        ColorModel cm1 = new ColorModel(ColorSpace.CS_BT709, 
                                        ColorRange.CR_MPEG,
                                        ColorPrimaries.CP_BT709,
                                        ColorTransferCharacteristic.CTC_BT709);
        PixelInfo pi1 = new PixelInfo(PixelFormat.PF_NV21, cm1);
        assertTrue(pi1.format() == PixelFormat.PF_NV21);
        assertTrue(pi1.space() == ColorSpace.CS_BT709);
        assertTrue(pi1.primaries() == ColorPrimaries.CP_BT709);
        assertTrue(pi1.range() == ColorRange.CR_MPEG);
        assertTrue(pi1.colorTransferCharacteristic() == ColorTransferCharacteristic.CTC_BT709);
        pi1.free();

        //
        PixelInfo pi2 = new PixelInfo(PixelFormat.PF_NV21, ColorSpace.CS_BT709, ColorRange.CR_MPEG);
        assertTrue(pi2.format() == PixelFormat.PF_NV21);
        assertTrue(pi2.space() == ColorSpace.CS_BT709);
        assertTrue(pi2.primaries() == ColorPrimaries.CP_UNSPECIFIED);
        assertTrue(pi2.range() == ColorRange.CR_MPEG);
        assertTrue(pi2.colorTransferCharacteristic() == ColorTransferCharacteristic.CTC_UNSPECIFIED);
        pi2.free();
    }
}
