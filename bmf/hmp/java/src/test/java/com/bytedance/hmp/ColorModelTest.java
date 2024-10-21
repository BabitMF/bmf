package com.bytedance.hmp;

import junit.framework.TestCase;

public class ColorModelTest extends TestCase {
    public void testColorModelCreate()
    {
        ColorModel cm0 = new ColorModel(ColorSpace.CS_BT709, 
                                        ColorRange.CR_MPEG,
                                        ColorPrimaries.CP_BT709,
                                        ColorTransferCharacteristic.CTC_BT709);
        assertTrue(cm0.space() == ColorSpace.CS_BT709);
        assertTrue(cm0.range() == ColorRange.CR_MPEG);
        assertTrue(cm0.primaries() == ColorPrimaries.CP_BT709);
        assertTrue(cm0.colorTransferCharacteristic() == ColorTransferCharacteristic.CTC_BT709);

        cm0.free();
    }
}
