package com.bmf.lite;

public class PropertyParam {
    public int propertyStatus;
    public Param param = null;
    public PropertyParam() {}
    public PropertyParam(int status, long paramData) {
        param = new Param(paramData);
        propertyStatus = status;
    }
    public void Free() {
        if (param != null) {
            param.Free();
        }
    }
    public long getPropertyParamNativePtr() {
        if (param != null) {
            return param.getNativePtr();
        } else {
            return 0l;
        }
    }

    public void setPropertyParam(Param paramData, int status) {
        param = paramData;
        propertyStatus = status;
        return;
    }
}
