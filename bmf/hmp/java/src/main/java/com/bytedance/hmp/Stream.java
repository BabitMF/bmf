package com.bytedance.hmp;


public class Stream extends Ptr {
    public class Guard extends Ptr{
        public Guard()
        {
            ptr = Api.stream_guard_create(Stream.this.ptr);
            own = true;
        }

        public void free()
        {
            if(own){
                Api.stream_guard_free(ptr);
            }
        }
    }

    // 
    public static Stream wrap(long ptr, boolean own)
    {
        return new Stream(ptr, own);
    }

    Stream(long ptr_, boolean own_)
    {
        ptr = ptr_;
        own = own_;
    }

    public Stream(DeviceType device_type, long flags)
    {
        ptr = Api.stream_make(device_type.getValue(), flags);
        own = true;
    }

    public void free()
    {
        if(own){
            Api.stream_free(ptr);
        }
    }

    //
    public boolean query()
    {
        return Api.stream_query(ptr);
    }

    public void synchronize()
    {
        Api.stream_synchronize(ptr);
    }

    public long handle()
    {
        return Api.stream_handle(ptr);
    }

    public DeviceType deviceType()
    {
        int v = Api.stream_device_type(ptr);
        return (DeviceType)EnumUtil.fromValue(DeviceType.class, v);
    }

    public int deviceIndex()
    {
        return Api.stream_device_index(ptr);
    }
	
    public static void setCurrent(Stream stream)
    {
        Api.stream_set_current(stream.ptr);
    }

    public static Stream current(DeviceType deviceType)
    {
        return wrap(Api.stream_current(deviceType.getValue()), true);
    }
}
