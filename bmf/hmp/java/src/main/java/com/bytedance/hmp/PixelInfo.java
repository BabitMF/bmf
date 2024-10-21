package com.bytedance.hmp;


public class PixelInfo extends Ptr{
    public static PixelInfo wrap(long ptr, boolean own)
    {
        return new PixelInfo(ptr, own);
    }

    private PixelInfo(long ptr_, boolean own_)
    {
        ptr = ptr_;
        own = own_;
    }

    public PixelInfo(PixelFormat format, ColorModel cm)
    {
        ptr = Api.pixel_info_make(format.getValue(), cm.getPtr());
        own = true;
    }

    public PixelInfo(PixelFormat format)
    {
        ptr = Api.pixel_info_make(format.getValue(), 
                                  ColorSpace.CS_UNSPECIFIED.getValue(),
                                  ColorRange.CR_UNSPECIFIED.getValue());
        own = true;
    }

    public PixelInfo(PixelFormat format, ColorSpace cs, ColorRange cr)
    {
        ptr = Api.pixel_info_make(format.getValue(), cs.getValue(), cr.getValue());
        own = true;
    }

    public void free()
    {
        if(own){
            Api.pixel_info_free(ptr);
        }
    }

    public PixelFormat format()
    {
        int v = Api.pixel_info_format(ptr);
        return (PixelFormat)EnumUtil.fromValue(PixelFormat.class, v);
    }

    public ColorSpace space()
    {
        int v = Api.pixel_info_space(ptr);
        return (ColorSpace)EnumUtil.fromValue(ColorSpace.class, v);
    }

    public ColorRange range()
    {
        int v = Api.pixel_info_range(ptr);
        return (ColorRange)EnumUtil.fromValue(ColorRange.class, v);
    }

    public ColorPrimaries primaries()
    {
        int v = Api.pixel_info_primaries(ptr);
        return (ColorPrimaries)EnumUtil.fromValue(ColorPrimaries.class, v);
    }

    public ColorTransferCharacteristic colorTransferCharacteristic()
    {
        int v = Api.pixel_info_primaries(ptr);
        return (ColorTransferCharacteristic)EnumUtil.fromValue(
            ColorTransferCharacteristic.class, v);
    }

    public ColorSpace inferSpace()
    {
        int v = Api.pixel_info_infer_space(ptr);
        return (ColorSpace)EnumUtil.fromValue(ColorSpace.class, v);
    }

    public ColorModel colorModel()
    {
        long cm = Api.pixel_info_color_model(ptr);
        return ColorModel.wrap(cm, false);
    }

    public boolean isRgbx()
    {
        return Api.pixel_info_is_rgbx(ptr);
    }

    public String toString()
    {
        return Api.pixel_info_stringfy(ptr);
    }

}
