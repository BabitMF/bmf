package com.bytedance.hmp;


public class ColorModel extends Ptr {
    public static ColorModel wrap(long ptr, boolean own)
    {
        return new ColorModel(ptr, own);
    }

    private ColorModel(long ptr_, boolean own_)
    {
        ptr = ptr_;
        own = own_;
    }

    public ColorModel(ColorSpace cs, ColorRange cr, 
                      ColorPrimaries cp, ColorTransferCharacteristic ctc)
    {
        ptr = Api.color_model_make(cs.getValue(), cr.getValue(), cp.getValue(), ctc.getValue());
        own = true;
    }

    public void free()
    {
        if(own){
            Api.color_model_free(ptr);
        }
    }

    public ColorSpace space()
    {
        int v = Api.color_model_space(ptr);
        return (ColorSpace)EnumUtil.fromValue(ColorSpace.class, v);
    }

    public ColorRange range()
    {
        int v = Api.color_model_range(ptr);
        return (ColorRange)EnumUtil.fromValue(ColorRange.class, v);
    }

    public ColorPrimaries primaries()
    {
        int v = Api.color_model_primaries(ptr);
        return (ColorPrimaries)EnumUtil.fromValue(ColorPrimaries.class, v);
    }

    public ColorTransferCharacteristic colorTransferCharacteristic()
    {
        int v = Api.color_model_ctc(ptr);
        return (ColorTransferCharacteristic)EnumUtil.fromValue(ColorTransferCharacteristic.class, v);
    }
}
