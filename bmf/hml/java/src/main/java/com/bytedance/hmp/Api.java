package com.bytedance.hmp;
import android.graphics.Bitmap;

public class Api {
    private static final String sJniLib = "_jhmp";
    static{
        System.loadLibrary(sJniLib); 
    }


    // Scalar
    static native long scalar(double v);
    static native long scalar(long v);
    static native long scalar(boolean v);
    static native void scalar_free(long ptr);

    // Device
    static native int device_count(int deviceType);
    static native long device_make(String dstr);
    static native long device_make(int deviceType, int index);
    static native void device_free(long ptr);
    static native int device_type(long ptr);
    static native int device_index(long ptr);
    static native String device_stringfy(long ptr);

    static native long device_guard_make(long device);
    static native void device_guard_free(long ptr);

    // Stream
    static native long stream_make(int device_type, long flags);
    static native void stream_free(long ptr);
    static native boolean stream_query(long ptr);
    static native void stream_synchronize(long ptr);
    static native long stream_handle(long ptr);
    static native int stream_device_type(long ptr);
    static native int stream_device_index(long ptr);
    static native void stream_set_current(long ptr);
    static native long stream_current(int device_type);

    static native long stream_guard_create(long stream);
    static native void stream_guard_free(long ptr);
    
    // Tensor
    static native long tensor_empty(long[] shape, int dtype, String device, boolean pinned_memory);
    static native long tensor_arange(long start, long end, long step, int dtype, String device, boolean pinned_memory);
    static native void tensor_free(long ptr);
    static native String tensor_stringfy(long ptr);

    static native void tensor_fill(long ptr, long scalar);
    static native boolean tensor_defined(long ptr);
    static native long tensor_dim(long ptr);
    static native long tensor_size(long ptr, long dim);
    static native long tensor_stride(long ptr, long dim);
    static native long tensor_nitems(long ptr);
    static native long tensor_itemsize(long ptr);
    static native long tensor_nbytes(long ptr);
    static native int tensor_dtype(long ptr);
    static native boolean tensor_is_contiguous(long ptr);
    static native int tensor_device_type(long ptr);
    static native int tensor_device_index(long ptr);
    static native long tensor_data_ptr(long ptr); 

    static native long tensor_clone(long ptr);
    static native long tensor_alias(long ptr);
    static native long tensor_view(long ptr, long[] shape);
    static native long tensor_reshape(long ptr, long[] shape);

    static native long tensor_slice(long ptr, long dim, long start, long end, long step);
    static native long tensor_select(long ptr, long dim, long index);
    static native long tensor_permute(long ptr, long[] dims);
    static native long tensor_squeeze(long ptr, long dim);
    static native long tensor_unsqueeze(long ptr, long dim);
    static native long tensor_to_device(long ptr, String device, boolean non_blocking);
    static native long tensor_to_dtype(long ptr, int dtype);
    static native void tensor_copy_from(long ptr, long from);

    static native long tensor_from_file(String fn, int dtype, long count, long offset);
    static native void tensor_to_file(long data, String fn);

    // ColorModel
    static native long color_model_make(int cs, int cr, int cp, int ctc);
    static native void color_model_free(long ptr);
    static native int color_model_space(long ptr);
    static native int color_model_range(long ptr);
    static native int color_model_primaries(long ptr);
    static native int color_model_ctc(long ptr);

    // PixelInfo
    static native long pixel_info_make(int format, long cm);
    static native long pixel_info_make(int format, int cs, int cr);
    static native void pixel_info_free(long ptr);

    static native int pixel_info_format(long ptr);
    static native int pixel_info_space(long ptr);
    static native int pixel_info_range(long ptr);
    static native int pixel_info_primaries(long ptr);
    static native int pixel_info_ctc(long ptr);
    static native int pixel_info_infer_space(long ptr);
    static native long pixel_info_color_model(long ptr);
    static native boolean pixel_info_is_rgbx(long ptr);
    static native String pixel_info_stringfy(long ptr);

    // PixelFormatDesc
    static native long pixel_format_desc_make(int format);
    static native void pixel_format_desc_free(long ptr);
    static native int pixel_format_desc_nplanes(long ptr);
    static native int pixel_format_desc_dtype(long ptr);
    static native int pixel_format_desc_format(long ptr);
    static native int pixel_format_desc_channels(long ptr, int plane);
    static native int pixel_format_desc_infer_width(long ptr, int width, int plane);
    static native int pixel_format_desc_infer_height(long ptr, int height, int plane);
    static native int pixel_format_desc_infer_nitems(long ptr, int width, int height);
    static native int pixel_format_desc_infer_nitems(long ptr, int width, int height, int plane);

    // Frame
    static native long frame_make(int width, int height, long pix_info, String device);
    static native long frame_make(long[] data, long pix_info);
    static native long frame_make(long[] data, int width, int height, long pix_info);
    static native long frame_make(Bitmap bitmap);
    static native void frame_free(long ptr);

    static native boolean frame_defined(long ptr);
    static native long frame_pix_info(long ptr);
    static native int frame_format(long ptr);
    static native int frame_width(long ptr);
    static native int frame_height(long ptr);
    static native int frame_dtype(long ptr);
    static native int frame_device_type(long ptr);
    static native int frame_device_index(long ptr);
    static native int frame_nplanes(long ptr);
    static native long frame_plane(long ptr, int p);
    static native long frame_to_device(long ptr, String device, boolean non_blocking);
    static native void frame_copy_from(long ptr, long from);
    static native long frame_clone(long ptr);
    static native long frame_crop(long ptr, int left, int top, int width, int height);
    static native long frame_to_image(long ptr, int cformat);
    static native long frame_from_image(long ptr, long pix_info);
    static native String frame_stringfy(long ptr);

    // Image
    static native long image_make(int width, int height, int channels, 
                                  int cformat, int dtype, String device, boolean pinned_memory);
    static native long image_make(long data, int cformat);
    static native long image_make(long data, int cformat, long cm);
    static native void image_free(long ptr);

    static native boolean image_defined(long ptr);
    static native int image_format(long ptr);
    static native void image_set_color_model(long ptr, long cm);
    static native long image_color_model(long ptr);
    static native int image_wdim(long ptr);
    static native int image_hdim(long ptr);
    static native int image_cdim(long ptr);
    static native int image_width(long ptr);
    static native int image_height(long ptr);
    static native int image_nchannels(long ptr);
    static native int image_dtype(long ptr);
    static native int image_device_type(long ptr);
    static native int image_device_index(long ptr);
    static native long image_data(long ptr);
    static native long image_to_device(long ptr, String device, boolean non_blocking);
    static native long image_to_dtype(long ptr, int dtype);
    static native void image_copy_from(long ptr, long from);
    static native long image_clone(long ptr);
    static native long image_crop(long ptr, int left, int top, int width, int height);
    static native String image_stringfy(long ptr);
}
