package com.bytedance.hmp;


public class Scalar extends Ptr{
    public static Scalar wrap(long ptr, boolean own)
    {
        return new Scalar(ptr, own);
    }

    Scalar(long ptr_, boolean own_)
    {
        ptr = ptr_;
        own = own_;
    }

    public Scalar(double v)
    {
        ptr = Api.scalar(v);
        own = true;
    }

    public Scalar(long v)
    {
        ptr = Api.scalar(v);
        own = true;
    }

    public Scalar(boolean v)
    {
        ptr = Api.scalar(v);
        own = true;
    }

    public void free()
    {
        if(own){
            Api.scalar_free(ptr);
        }
    }
}
