package com.bmf.lite;
import com.bmf.lite.common.ErrorCode;
import com.bmf.lite.common.SoLoader;
public class Param {
    private static final String TAG = "BmfParam";
    private long nativePtr = 0l;
    public Param() {}

    public Param(long ptr) {
        if (nativePtr != 0l) {
            nativeReleaseAlgorithmParam(nativePtr);
        }
        nativePtr = ptr;
    }

    public void Free() {
        if (nativePtr == 0l) {
            return;
        }
        nativeReleaseAlgorithmParam(nativePtr);
    }

    public int init() {
        if (!SoLoader.getInstance().isSoInitialized()) {
            return ErrorCode.LOAD_SO_FAIL;
        }
        nativePtr = nativeCreateAlgorithmParam();
        if (nativePtr == 0l) {
            return ErrorCode.CREATE_JNI_RESOURCE_FAIL;
        }
        return 0;
    }

    public long getNativePtr() {
        if (!SoLoader.getInstance().isSoInitialized()) {
            return ErrorCode.LOAD_SO_FAIL;
        }
        return nativePtr;
    }
    public void setNativePtr(long ptr) { nativePtr = ptr; }

    public boolean hasKey(String key) {
        if (!SoLoader.getInstance().isSoInitialized()) {
            return false;
        }
        if (nativePtr == 0l) {
            return false;
        }
        return nativeHasKey(nativePtr, key);
    }

    public int eraseKey(String key) {
        if (!SoLoader.getInstance().isSoInitialized()) {
            return ErrorCode.LOAD_SO_FAIL;
        }
        if (nativePtr == 0l) {
            return ErrorCode.LOAD_SO_FAIL;
        }
        return nativeEraseKey(nativePtr, key);
    }

    public int setInt(String key, int value) {
        if (!SoLoader.getInstance().isSoInitialized()) {
            return ErrorCode.LOAD_SO_FAIL;
        }
        if (nativePtr == 0l) {
            return ErrorCode.JNI_RESOURCE_NOT_INIT;
        }
        return nativeSetInt(nativePtr, key, value);
    }

    public int setLong(String key, long value) {
        if (!SoLoader.getInstance().isSoInitialized()) {
            return ErrorCode.LOAD_SO_FAIL;
        }
        if (nativePtr == 0l) {
            return ErrorCode.JNI_RESOURCE_NOT_INIT;
        }
        return nativeSetLong(nativePtr, key, value);
    }

    public int setFloat(String key, float value) {
        if (!SoLoader.getInstance().isSoInitialized()) {
            return ErrorCode.LOAD_SO_FAIL;
        }
        if (nativePtr == 0l) {
            return ErrorCode.JNI_RESOURCE_NOT_INIT;
        }
        return nativeSetFloat(nativePtr, key, value);
    }

    public int setDouble(String key, double value) {
        if (!SoLoader.getInstance().isSoInitialized()) {
            return ErrorCode.LOAD_SO_FAIL;
        }
        if (nativePtr == 0l) {
            return ErrorCode.JNI_RESOURCE_NOT_INIT;
        }
        return nativeSetDouble(nativePtr, key, value);
    }

    public int setString(String key, String value) {
        if (!SoLoader.getInstance().isSoInitialized()) {
            return ErrorCode.LOAD_SO_FAIL;
        }
        if (nativePtr == 0l) {
            return ErrorCode.JNI_RESOURCE_NOT_INIT;
        }
        return nativeSetString(nativePtr, key, value);
    }

    public int setIntList(String key, int[] value) {
        if (!SoLoader.getInstance().isSoInitialized()) {
            return ErrorCode.LOAD_SO_FAIL;
        }
        if (nativePtr == 0l) {
            return ErrorCode.JNI_RESOURCE_NOT_INIT;
        }
        return nativeSetIntList(nativePtr, key, value);
    }

    public int setLongList(String key, long[] value) {
        if (!SoLoader.getInstance().isSoInitialized()) {
            return ErrorCode.LOAD_SO_FAIL;
        }
        if (nativePtr == 0l) {
            return ErrorCode.JNI_RESOURCE_NOT_INIT;
        }
        return nativeSetLongList(nativePtr, key, value);
    }

    public int setFloatList(String key, float[] value) {
        if (!SoLoader.getInstance().isSoInitialized()) {
            return ErrorCode.LOAD_SO_FAIL;
        }
        if (nativePtr == 0l) {
            return ErrorCode.JNI_RESOURCE_NOT_INIT;
        }
        return nativeSetFloatList(nativePtr, key, value);
    }

    public int setDoubleList(String key, double[] value) {
        if (!SoLoader.getInstance().isSoInitialized()) {
            return ErrorCode.LOAD_SO_FAIL;
        }
        if (nativePtr == 0l) {
            return ErrorCode.JNI_RESOURCE_NOT_INIT;
        }
        return nativeSetDoubleList(nativePtr, key, value);
    }

    //    int setString(String key,String value){
    //        if (!SoLoader.getInstance().isSoInitialized()) {
    //            return ErrorCode.LOAD_SO_FAIL;
    //        }
    //        if(nativePtr == 0l){
    //            return ErrorCode.JNI_RESOURCE_NOT_INIT;
    //        }
    //        return nativeSetString(nativePtr,key,value);
    //    }

    public int getInt(String key) {
        if (!SoLoader.getInstance().isSoInitialized()) {
            return ErrorCode.LOAD_SO_FAIL;
        }
        if (nativePtr == 0l) {
            return ErrorCode.JNI_RESOURCE_NOT_INIT;
        }
        return nativeGetInt(nativePtr, key);
    }

    public long getLong(String key) {
        if (!SoLoader.getInstance().isSoInitialized()) {
            return ErrorCode.LOAD_SO_FAIL;
        }
        if (nativePtr == 0l) {
            return ErrorCode.JNI_RESOURCE_NOT_INIT;
        }
        return nativeGetLong(nativePtr, key);
    }

    public float getFloat(String key) {
        if (!SoLoader.getInstance().isSoInitialized()) {
            return ErrorCode.LOAD_SO_FAIL;
        }
        if (nativePtr == 0l) {
            return ErrorCode.JNI_RESOURCE_NOT_INIT;
        }
        return nativeGetFloat(nativePtr, key);
    }

    public double getDoublbe(String key) {
        if (!SoLoader.getInstance().isSoInitialized()) {
            return ErrorCode.LOAD_SO_FAIL;
        }
        if (nativePtr == 0l) {
            return ErrorCode.JNI_RESOURCE_NOT_INIT;
        }
        return nativeGetDouble(nativePtr, key);
    }

    public String getString(String key) {
        if (!SoLoader.getInstance().isSoInitialized()) {
            return "";
        }
        if (nativePtr == 0l) {
            return "";
        }
        return nativeGetString(nativePtr, key);
    }
    public int[] getIntList(String key) {
        if (!SoLoader.getInstance().isSoInitialized()) {
            return null;
        }
        if (nativePtr == 0l) {
            return null;
        }
        return nativeGetIntList(nativePtr, key);
    }
    public long[] getLongList(String key) {
        if (!SoLoader.getInstance().isSoInitialized()) {
            return null;
        }
        if (nativePtr == 0l) {
            return null;
        }
        return nativeGetLongList(nativePtr, key);
    }
    public float[] getFloatList(String key) {
        if (!SoLoader.getInstance().isSoInitialized()) {
            return null;
        }
        if (nativePtr == 0l) {
            return null;
        }
        return nativeGetFloatList(nativePtr, key);
    }

    public double[] getDoublbeList(String key) {
        if (!SoLoader.getInstance().isSoInitialized()) {
            return null;
        }
        if (nativePtr == 0l) {
            return null;
        }
        return nativeGetDoubleList(nativePtr, key);
    }

    public String[] getStringList(String key) {
        if (!SoLoader.getInstance().isSoInitialized()) {
            return null;
        }
        if (nativePtr == 0l) {
            return null;
        }
        return nativeGetStringList(nativePtr, key);
    }
    private native long nativeCreateAlgorithmParam();

    private native void nativeReleaseAlgorithmParam(long p);

    private native boolean nativeHasKey(long p, String key);
    private native int nativeEraseKey(long p, String key);
    private native int nativeSetInt(long p, String key, int value);
    private native int nativeSetLong(long p, String key, long value);
    private native int nativeSetFloat(long p, String key, float value);
    private native int nativeSetDouble(long p, String key, double value);
    private native int nativeSetString(long p, String key, String value);
    private native int nativeSetIntList(long p, String key, int[] value);
    private native int nativeSetLongList(long p, String key, long[] value);
    private native int nativeSetFloatList(long p, String key, float[] value);
    private native int nativeSetDoubleList(long p, String key, double[] value);
    private native int nativeSetStringList(long p, String key, String[] value);
    private native int nativeGetInt(long p, String key);
    private native long nativeGetLong(long p, String key);
    private native float nativeGetFloat(long p, String key);
    private native double nativeGetDouble(long p, String key);
    private native String nativeGetString(long p, String key);
    private native int[] nativeGetIntList(long p, String key);
    private native long[] nativeGetLongList(long p, String key);
    private native float[] nativeGetFloatList(long p, String key);
    private native double[] nativeGetDoubleList(long p, String key);
    private native String[] nativeGetStringList(long p, String key);
}
