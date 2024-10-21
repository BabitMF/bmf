package com.bytedance.hmp;

public class Tensor extends Ptr {
    public static Tensor wrap(long ptr, boolean own)
    {
        return new Tensor(ptr, own);
    }

    Tensor(long ptr_, boolean own_)
    {
        ptr = ptr_;
        own = own_;
    }

    public void free()
    {
        if(own){
            Api.tensor_free(ptr);
        }
    }

    public String toString()
    {
        return Api.tensor_stringfy(ptr);
    }

    public void fill(long value)
    {
        Scalar v = new Scalar(value);
        Api.tensor_fill(ptr, v.getPtr());
        v.free();
    }

    public void fill(double value)
    {
        Scalar v = new Scalar(value);
        Api.tensor_fill(ptr, v.getPtr());
        v.free();
    }

    public void fill(boolean value)
    {
        Scalar v = new Scalar(value);
        Api.tensor_fill(ptr, v.getPtr());
        v.free();
    }

    public boolean defined()
    {
        return Api.tensor_defined(ptr);
    }

    public long dim()
    {
        return Api.tensor_dim(ptr);
    }

    public long size(long dim)
    {
        return Api.tensor_size(ptr, dim);
    }

    public long stride(long dim)
    {
        return Api.tensor_stride(ptr, dim);
    }

    public long nitems()
    {
        return Api.tensor_nitems(ptr);
    }

    public long itemsize()
    {
        return Api.tensor_itemsize(ptr);
    }

    public long nbytes()
    {
        return Api.tensor_nbytes(ptr);
    }

    public ScalarType dtype()
    {
        int v = Api.tensor_dtype(ptr);
        return (ScalarType)EnumUtil.fromValue(ScalarType.class, v);
    }

    public boolean isContiguous()
    {
        return Api.tensor_is_contiguous(ptr);
    }

    public DeviceType deviceType()
    {
        int v = Api.tensor_device_type(ptr);
        return (DeviceType)EnumUtil.fromValue(DeviceType.class, v);
    }

    public int deviceIndex()
    {
        return Api.tensor_device_index(ptr);
    }

    public long dataPtr()
    {
        return Api.tensor_data_ptr(ptr);
    }

    public Tensor clone()
    {
        long nptr = Api.tensor_clone(ptr);
        return wrap(nptr, true);
    }

    public Tensor alias()
    {
        long nptr = Api.tensor_alias(ptr);
        return wrap(nptr, true);
    }

    public Tensor view(long[] shape)
    {
        long nptr = Api.tensor_view(ptr, shape);
        return wrap(nptr, true);
    }

    public Tensor reshape(long[] shape)
    {
        long nptr = Api.tensor_reshape(ptr, shape);
        return wrap(nptr, true);
    }

    public Tensor slice(long dim, long start, long end, long step)
    {
        long nptr = Api.tensor_slice(ptr, dim, start, end, step);
        return wrap(nptr, true);
    }

    public Tensor select(long dim, long index)
    {
        long nptr = Api.tensor_select(ptr, dim, index);
        return wrap(nptr, true);
    }

    public Tensor permute(long[] dims)
    {
        long nptr = Api.tensor_permute(ptr, dims);
        return wrap(nptr, true);
    }

    public Tensor squeeze(long dim)
    {
        long nptr = Api.tensor_squeeze(ptr, dim);
        return wrap(nptr, true);
    }

    public Tensor unsqueeze(long dim)
    {
        long nptr = Api.tensor_unsqueeze(ptr, dim);
        return wrap(nptr, true);
    }

    public Tensor to(String device, boolean nonBlocking)
    {
        long nptr = Api.tensor_to_device(ptr, device, nonBlocking);
        return wrap(nptr, true);
    }

    public Tensor to(Device device, boolean nonBlocking)
    {
        return to(device.toString(), nonBlocking);
    }

    public Tensor to(ScalarType dtype)
    {
        long nptr = Api.tensor_to_dtype(ptr, dtype.getValue());
        return wrap(nptr, true);
    }

    public void copyFrom(Tensor from)
    {
        Api.tensor_copy_from(ptr, from.getPtr());
    }


    ///
    public static Tensor empty(long[] shape, ScalarType dtype, String device, boolean pinned_memory)
    {
        long ptr = Api.tensor_empty(shape, dtype.getValue(), device, pinned_memory);
        return wrap(ptr, true);
    }

    public static Tensor empty(long[] shape, ScalarType dtype, Device device, boolean pinned_memory)
    {
        return empty(shape, dtype, device.toString(), pinned_memory);
    }

    public static Tensor arange(long start, long end, long step, ScalarType dtype, String device, boolean pinned_memory)
    {
        long ptr = Api.tensor_arange(start, end, step, dtype.getValue(), device, pinned_memory);
        return wrap(ptr, true);
    }

    public static Tensor arange(long start, long end, long step, ScalarType dtype, Device device, boolean pinned_memory)
    {
        return arange(start, end, step, dtype, device.toString(), pinned_memory);
    }

    public static Tensor fromFile(String fn, ScalarType dtype, long count, long offset)
    {
        long ptr = Api.tensor_from_file(fn, dtype.getValue(), count, offset);
        return wrap(ptr, true);
    }

    public void toFile(String fn)
    {
        Api.tensor_to_file(ptr, fn);
    }

}
