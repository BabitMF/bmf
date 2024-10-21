package com.bytedance.hmp;


public class Image extends Ptr {
    public static Image wrap(long ptr, boolean own)
    {
        return new Image(ptr, own);
    }

    private Image(long ptr_, boolean own_)
    {
        ptr = ptr_;
        own = own_;
    }
	
    public Image(int width, int height, int channels, ChannelFormat cformat, 
                ScalarType dtype, String device, boolean pinnedMemory)
    {
        ptr = Api.image_make(width, height, channels, cformat.getValue(),
                             dtype.getValue(), device, pinnedMemory);
        own = true;
    }

    public Image(Tensor data, ChannelFormat cformat)
    {
        ptr = Api.image_make(data.getPtr(), cformat.getValue());
        own = true;
    }

    public Image(Tensor data, ChannelFormat cformat, ColorModel cm)
    {
        ptr = Api.image_make(data.getPtr(), cformat.getValue(), cm.getPtr());
        own = true;
    }

    public void free()
    {
        if(own){
            Api.image_free(ptr);
        }
    }

    public boolean defined()
    {
        return Api.image_defined(ptr);
    }

    public void setColorModel(ColorModel cm)
    {
        Api.image_set_color_model(ptr, cm.getPtr());
    }

    public ColorModel colorModel()
    {
        return ColorModel.wrap(Api.image_color_model(ptr), false);
    }

    public int wdim()
    {
        return Api.image_wdim(ptr);
    }

    public int hdim()
    {
        return Api.image_hdim(ptr);
    }

    public int cdim()
    {
        return Api.image_cdim(ptr);
    }

    public int width()
    {
        return Api.image_width(ptr);
    }

    public int height()
    {
        return Api.image_height(ptr);
    }

    public int nchannels()
    {
        return Api.image_nchannels(ptr);
    }

    public ScalarType dtype()
    {
        int v = Api.image_dtype(ptr);
        return (ScalarType)EnumUtil.fromValue(ScalarType.class, v);
    }

    public DeviceType deviceType()
    {
        int v = Api.image_device_type(ptr);
        return (DeviceType)EnumUtil.fromValue(DeviceType.class, v);
    }

    public int deviceIndex()
    {
        return Api.image_device_index(ptr);
    }

    public Tensor data()
    {
        long p = Api.image_data(ptr);
        return Tensor.wrap(ptr, false);
    }

    public Image to(String device, boolean nonBlocking)
    {
        return Image.wrap(Api.image_to_device(ptr, device, nonBlocking), true);
    }

    public Image to(Device device, boolean nonBlocking)
    {
        return to(device.toString(), nonBlocking);
    }

    public void copyFrom(Image from)
    {
        Api.image_copy_from(ptr, from.getPtr());
    }

    public Image clone()
    {
        return Image.wrap(Api.image_clone(ptr), true);
    }

    public Image crop(int left, int top, int width, int height)
    {
        return Image.wrap(Api.image_crop(ptr, left, top, width, height), true);
    }

    public String toString()
    {
        return Api.image_stringfy(ptr);
    }



}
