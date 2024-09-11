package com.bmf.lite;

import androidx.annotation.Keep;

import com.bmf.lite.common.ErrorCode;

import com.bmf.lite.common.SoLoader;

import java.util.List;

public class AlgorithmInterface {
    private static final String TAG = "BmfAlgorithm";
    private long nativeAlgorithmPtr = 0l;
    int bmfMulFrameSize = 0;

    public AlgorithmInterface() {}

    public int init() {
        if (!SoLoader.getInstance().isSoInitialized()) {
            return ErrorCode.LOAD_SO_FAIL;
        }
        nativeAlgorithmPtr = nativeCreateAlgorithm();
        if (nativeAlgorithmPtr == 0l) {
            return ErrorCode.CREATE_JNI_RESOURCE_FAIL;
        }
        return 0;
    }

    public void Free() {
        if (nativeAlgorithmPtr == 0l) {
            return;
        }
        nativeReleaseAlgorithm(nativeAlgorithmPtr);
    }

    public int setParam(Param param) {
        if (!SoLoader.getInstance().isSoInitialized()) {
            return ErrorCode.LOAD_SO_FAIL;
        }
        if (nativeAlgorithmPtr == 0l) {
            return ErrorCode.JNI_RESOURCE_NOT_INIT;
        }
        int result =
            nativeSetAlgorithmParam(nativeAlgorithmPtr, param.getNativePtr());
        return result;
    }

    public int processVideoFrame(VideoFrame videoframe, Param param) {
        if (!SoLoader.getInstance().isSoInitialized()) {
            return ErrorCode.LOAD_SO_FAIL;
        }
        if (nativeAlgorithmPtr == 0l) {
            return ErrorCode.JNI_RESOURCE_NOT_INIT;
        }
        if (videoframe == null || param == null) {
            return ErrorCode.INVALID_PARAMETER;
        }
        int result = nativeProcessVideoFrame(nativeAlgorithmPtr,
                                             videoframe.getNativePtr(),
                                             param.getNativePtr());
        return result;
    }

    public VideoFrameOutput getVideoFrameOutput() {
        if (!SoLoader.getInstance().isSoInitialized()) {
            return null;
        }
        if (nativeAlgorithmPtr == 0l) {
            return null;
        }

        VideoFrameOutput bmfVideoFrameOutput =
            nativeGetVideoFrameOutput(nativeAlgorithmPtr);
        // Logging.d(TAG, "bmfAlgorithm java getVideoFrameOutput
        // result"+bmfVideoFrameOutput.outputStatus);
        return bmfVideoFrameOutput;
    }

    public int processMultiVideoFrame(List<VideoFrame> videoframes,
                                      List<Param> params) {
        if (!SoLoader.getInstance().isSoInitialized()) {
            return ErrorCode.LOAD_SO_FAIL;
        }
        if (nativeAlgorithmPtr == 0l) {
            return ErrorCode.JNI_RESOURCE_NOT_INIT;
        }
        if (videoframes.size() != params.size() || params.size() <= 0 ||
            videoframes.size() <= 0) {
            return ErrorCode.INVALID_PARAMETER;
        }

        bmfMulFrameSize = 0;
        for (int i = 0; i < videoframes.size(); i++) {
            int result = nativeProcessVideoFrame(
                nativeAlgorithmPtr, videoframes.get(i).getNativePtr(),
                params.get(i).getNativePtr());
            if (result != 0) {
                return result;
            }
            bmfMulFrameSize++;
        }
        return 0;
    }

    public VideoFrameOutput[] getMultiVideoFrameOutput() {
        if (!SoLoader.getInstance().isSoInitialized()) {
            return null;
        }
        if (nativeAlgorithmPtr == 0l) {
            return null;
        }
        VideoFrameOutput[] bmfVideoFrameOutputs =
            new VideoFrameOutput[bmfMulFrameSize];
        for (int i = 0; i < bmfMulFrameSize; i++) {
            bmfVideoFrameOutputs[i] =
                nativeGetVideoFrameOutput(nativeAlgorithmPtr);
        }
        return bmfVideoFrameOutputs;
    }

    public PropertyParam getProcessProperty() {
        if (!SoLoader.getInstance().isSoInitialized()) {
            return null;
        }
        if (nativeAlgorithmPtr == 0l) {
            return null;
        }
        PropertyParam ropertyParam =
            nativeGetProcessProperty(nativeAlgorithmPtr);
        return ropertyParam;
    }

    public int setInputProperty(Param param) {
        if (!SoLoader.getInstance().isSoInitialized()) {
            return ErrorCode.LOAD_SO_FAIL;
        }
        if (nativeAlgorithmPtr == 0l) {
            return ErrorCode.JNI_RESOURCE_NOT_INIT;
        }
        if (param == null) {
            return ErrorCode.INVALID_PARAMETER;
        }
        return nativeSetInputProperty(nativeAlgorithmPtr, param.getNativePtr());
    }

    public PropertyParam getOutputProperty() {
        if (!SoLoader.getInstance().isSoInitialized()) {
            return null;
        }
        if (nativeAlgorithmPtr == 0l) {
            return null;
        }
        PropertyParam ropertyParam =
            nativeGetOutputProperty(nativeAlgorithmPtr);
        return ropertyParam;
    }

    private native long nativeCreateAlgorithm();

    private native void nativeReleaseAlgorithm(long native_alg_ptr);

    private native int nativeSetAlgorithmParam(long native_alg_ptr,
                                               long native_init_param_ptr);

    private native int nativeProcessVideoFrame(long native_alg_ptr,
                                               long native_video_frame_ptr,
                                               long native_process_param_ptr);

    @Keep
    private native VideoFrameOutput
    nativeGetVideoFrameOutput(long native_alg_ptr);

    @Keep
    private native PropertyParam nativeGetProcessProperty(long native_alg_ptr);

    @Keep
    private native PropertyParam nativeGetOutputProperty(long native_alg_ptr);

    private native int nativeSetInputProperty(long native_alg_ptr,
                                              long native_param_ptr);
}