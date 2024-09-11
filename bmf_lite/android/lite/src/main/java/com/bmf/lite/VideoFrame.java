package com.bmf.lite;
import com.bmf.lite.common.SoLoader;
import com.bmf.lite.common.ErrorCode;
public class VideoFrame {

    private static final String TAG = "BmfVideoFrame";
    public int pixelFormat;
    private long nativePtr = 0l;
    public VideoFrame() {}
    public VideoFrame(long ptr) {
        if (nativePtr != 0l) {
            nativeReleaseVideoFrame(nativePtr);
        }
        nativePtr = ptr;
    }
    public void Free() {
        if (nativePtr == 0l) {
            return;
        }
        nativeReleaseVideoFrame(nativePtr);
    }

    public int init() {
        if (!SoLoader.getInstance().isSoInitialized()) {
            return ErrorCode.LOAD_SO_FAIL;
        }
        nativePtr = nativeCreateVideoFrame();
        if (nativePtr == 0l) {
            return ErrorCode.CREATE_JNI_RESOURCE_FAIL;
        }
        return 0;
    }

    public int init(int textureId, int width, int height) {
        if (!SoLoader.getInstance().isSoInitialized()) {
            return ErrorCode.LOAD_SO_FAIL;
        }
        nativePtr = nativeCreateTextureVideoFrame(textureId, width, height);
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

    public int getTextureId() {
        if (!SoLoader.getInstance().isSoInitialized()) {
            return ErrorCode.LOAD_SO_FAIL;
        }
        if (nativePtr == 0l) {
            return ErrorCode.JNI_RESOURCE_NOT_INIT;
        }
        return nativeGetTextureId(nativePtr);
    }
    public int getWidth() {
        if (!SoLoader.getInstance().isSoInitialized()) {
            return ErrorCode.LOAD_SO_FAIL;
        }
        if (nativePtr == 0l) {
            return ErrorCode.JNI_RESOURCE_NOT_INIT;
        }
        return nativeGetWidth(nativePtr);
    }
    public int getHeight() {
        if (!SoLoader.getInstance().isSoInitialized()) {
            return ErrorCode.LOAD_SO_FAIL;
        }
        if (nativePtr == 0l) {
            return ErrorCode.JNI_RESOURCE_NOT_INIT;
        }
        return nativeGetHeight(nativePtr);
    }
    public void setPixelFormat(int pixFormat) { pixelFormat = pixFormat; }
    private native int nativeSetPixelFormat(long p, int pixFormat);
    private native int nativeGetTextureId(long p);
    private native int nativeGetWidth(long p);
    private native int nativeGetHeight(long p);
    private native long nativeCreateVideoFrame();
    private native long nativeCreateTextureVideoFrame(int textureId, int width,
                                                      int height);
    private native void nativeReleaseVideoFrame(long p);
}
