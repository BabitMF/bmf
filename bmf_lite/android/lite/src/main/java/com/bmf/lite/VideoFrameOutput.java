package com.bmf.lite;

public class VideoFrameOutput {
    public int outputStatus;
    public VideoFrame frame = null;
    public Param param = null;
    public VideoFrameOutput() {}
    public VideoFrameOutput(int status, long frameData, long paramData) {
        outputStatus = status;
        frame = new VideoFrame(frameData);
        param = new Param(paramData);
    }
    public void Free() {
        if (frame != null) {
            frame.Free();
        }
        if (param != null) {
            param.Free();
        }
    }
}
