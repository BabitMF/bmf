package com.bytedance.hmp;


public class Device extends Ptr {
    public class Guard extends Ptr{
        public Guard()
        {
            this.ptr = Api.device_guard_make(Device.this.ptr);
            this.own = true;
        }

        public void free()
        {
            if(this.own){
                Api.device_guard_free(this.ptr);
            }
        }
    }

    //
    public static Device wrap(long ptr, boolean own)
    {
        return new Device(ptr, own);
    }

    private Device(long ptr_, boolean own_)
    {
        ptr = ptr_;
        own = own_;
    }

    public Device(){
        ptr = Api.device_make("");
        own = true;
    }

    public Device(String device){
        ptr = Api.device_make(device);
        own = true;
    }

    public Device(DeviceType device, int index){
        ptr = Api.device_make(device.getValue(), index);
        own = true;
    }

    public void free()
    {
        if(own){
            Api.device_free(ptr);
        }
    }

    public boolean equals(Object other){
        if(this == other){
            return true;
        }

        if(other instanceof Device){
            Device dother = (Device)other;
            return type() == dother.type() && index() == dother.index();
        }
        else {
            return false;
        }
    }

    public DeviceType type()
    {
        int v = Api.device_type(ptr);
        return (DeviceType)EnumUtil.fromValue(DeviceType.class, v);
    }

    public int index()
    {
        return Api.device_index(ptr);
    }

    public String toString()
    {
        return Api.device_stringfy(ptr);
    }

    //
    public static int count(DeviceType deviceType)
    {
        return Api.device_count(deviceType.getValue());
    }

    public static boolean hasCuda()
    {
        return count(DeviceType.kCUDA) > 0;
    }
}