package com.bytedance.hmp;
import android.graphics.Bitmap;


public class Frame extends Ptr {
    public static Frame wrap(long ptr, boolean own)
    {
        return new Frame(ptr, own);
    }

    private Frame(long ptr_, boolean own_)
    {
        ptr = ptr_;
        own = own_;
    }
	
    public Frame(int width, int height, PixelInfo pix_info, String device)
    {
        ptr = Api.frame_make(width, height, pix_info.getPtr(), device);
        own = true;
    }

    public Frame(int width, int height, PixelInfo pix_info, Device device)
    {
        ptr = Api.frame_make(width, height, pix_info.getPtr(), device.toString());
        own = true;
    }

    public Frame(Tensor[] data, PixelInfo pix_info)
    {
        long[] dptrs = new long[data.length];
        for(int i = 0; i < data.length; i++){
            dptrs[i] = data[i].getPtr();
        }
        ptr = Api.frame_make(dptrs, pix_info.getPtr());
        own = true;
    }

    public Frame(Bitmap bitmap)
    {
        ptr = Api.frame_make(bitmap);
        own = true;
    }

    public Frame(Tensor[] data, int width, int height, PixelInfo pix_info)
    {
        long[] dptrs = new long[data.length];
        for(int i = 0; i < data.length; i++){
            dptrs[i] = data[i].getPtr();
        }
        ptr = Api.frame_make(dptrs, width, height, pix_info.getPtr());
        own = true;
    }

    public void free()
    {
        if(own){
            Api.frame_free(ptr);
        }
    }

    public boolean defined()
    {
        return Api.frame_defined(ptr);
    }

    public PixelInfo pixInfo()
    {
        return PixelInfo.wrap(Api.frame_pix_info(ptr), false);
    }

    public PixelFormat format()
    {
        int v = Api.frame_format(ptr);
        return (PixelFormat)EnumUtil.fromValue(PixelFormat.class, v);
    }

    public int width()
    {
        return Api.frame_width(ptr);
    }

    public int height()
    {
        return Api.frame_height(ptr);
    }

    public ScalarType dtype()
    {
        int v = Api.frame_dtype(ptr);
        return (ScalarType)EnumUtil.fromValue(ScalarType.class, v);
    }

    public DeviceType deviceType()
    {
        int v = Api.frame_device_type(ptr);
        return (DeviceType)EnumUtil.fromValue(DeviceType.class, v);
    }

    public int deviceIndex()
    {
        return Api.frame_device_index(ptr);
    }

    public int nplanes()
    {
        return Api.frame_nplanes(ptr);
    }

    public Tensor plane(int p)
    {
        long data = Api.frame_plane(ptr, p);
        return Tensor.wrap(data, false);
    }

    public Frame to(String device, boolean nonBlocking)
    {
        return Frame.wrap(Api.frame_to_device(ptr, device, nonBlocking), true);
    }

    public Frame to(Device device, boolean nonBlocking)
    {
        return to(device.toString(), nonBlocking);
    }

    public void copyFrom(Frame from)
    {
        Api.frame_copy_from(ptr, from.getPtr());
    }

    public Frame clone()
    {
        return Frame.wrap(Api.frame_clone(ptr), true);
    }

    public Frame crop(int left, int top, int width, int height)
    {
        return Frame.wrap(Api.frame_crop(ptr, left, top, width, height), true);
    }

    public Image toImage(ChannelFormat cformat)
    {
        return Image.wrap(Api.frame_to_image(ptr, cformat.getValue()), true);
    }

    public String toString()
    {
        return Api.frame_stringfy(ptr);
    }

    public static Frame fromImage(Image image, PixelInfo pixInfo)
    {
        return Frame.wrap(Api.frame_from_image(image.getPtr(), pixInfo.getPtr()), true);
    }

}
