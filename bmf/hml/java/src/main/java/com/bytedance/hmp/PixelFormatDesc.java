package com.bytedance.hmp;


public class PixelFormatDesc extends Ptr{
    public static PixelFormatDesc wrap(long ptr, boolean own)
    {
        return new PixelFormatDesc(ptr, own);
    }

    private PixelFormatDesc(long ptr_, boolean own_)
    {
        ptr = ptr_;
        own = own_;
    }

    public PixelFormatDesc(PixelFormat format)
    {
        ptr = Api.pixel_format_desc_make(format.getValue());
        own = true;
    }

    public void free()
    {
        if(own){
            Api.pixel_format_desc_free(ptr);
        }
    }

    public int nplanes()
    {
        return Api.pixel_format_desc_nplanes(ptr);
    }

    public ScalarType dtype()
    {
        int v = Api.pixel_format_desc_dtype(ptr);
        return (ScalarType)EnumUtil.fromValue(ScalarType.class, v);
    }

    public int format()
    {
        return Api.pixel_format_desc_format(ptr);
    }

    public int channels(int plane)
    {
        return Api.pixel_format_desc_channels(ptr, plane);
    }

    public int infer_width(int width, int plane)
    {
        return Api.pixel_format_desc_infer_width(ptr, width, plane);
    }

    public int infer_height(int height, int plane)
    {
        return Api.pixel_format_desc_infer_height(ptr, height, plane);
    }

    public int infer_nitems(int width, int height)
    {
        return Api.pixel_format_desc_infer_nitems(ptr, width, height);
    }

    public int infer_nitems(int width, int height, int plane)
    {
        return Api.pixel_format_desc_infer_nitems(ptr, width, height, plane);
    }
}
