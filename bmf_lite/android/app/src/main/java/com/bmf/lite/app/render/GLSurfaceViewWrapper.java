package com.bmf.lite.app.render;

import android.content.Context;
import android.graphics.SurfaceTexture;
import android.opengl.GLSurfaceView;
import android.util.AttributeSet;
import android.util.Log;
import android.view.SurfaceHolder;
import android.graphics.Bitmap;
public class GLSurfaceViewWrapper
    extends GLSurfaceView implements SurfaceTexture.OnFrameAvailableListener {
    private String TAG = "bmf-demo-app GLSurfaceViewWrapper";
    private GLSurfaceViewRender videoRender = null;
    private int videoWidth = 1080;
    private int videoHeight = 1920;
    private int wndWidth = 1080;
    private int wndHeight = 1920;
    private boolean initStatus = false;
    public GLSurfaceViewWrapper(Context context) {
        super(context);
        setEGLContextClientVersion(3);
    }

    public GLSurfaceViewWrapper(Context context, AttributeSet attrs) {
        super(context, attrs);
        setEGLContextClientVersion(3);
    }

    @Override
    public void onFrameAvailable(SurfaceTexture surfaceTexture) {
        Log.i(TAG, "surface onFrame avilable.");
        this.requestRender();
    }
    @Override
    public void surfaceChanged(SurfaceHolder holder, int format, int w, int h) {
        super.surfaceChanged(holder, format, w, h);
        wndWidth = w;
        wndHeight = h;
        if (videoRender != null) {
            Log.i(TAG, "mSurfaceViewWidth surfaceChanged,width " + w +
                           " height " + h);
            videoRender.setRenderWndSize(w, h);
        }
    }

    public void setRenderImgSize(int width, int height) {
        videoWidth = width;
        videoHeight = height;
        if (videoRender != null) {
            videoRender.setRenderImgSize(videoWidth, videoHeight);
        }
    }

    public int getWndWidth() { return wndWidth; }

    public int getWndHeight() { return wndHeight; }

    public void initRender() {
        if (initStatus) {
            return;
        }
        videoRender = new GLSurfaceViewRender();
        videoRender.setRenderImgSize(videoWidth, videoHeight);
        setRenderer(videoRender);
        initStatus = true;
    }
    public void setOesStatus(boolean isOes) {
        if (videoRender != null) {
            videoRender.setOesStatus(isOes);
        }
    }

    public void switchAlgorithm(int algType) {
        if (videoRender != null) {
            videoRender.switchAlgorithm(algType);
        }
    }

    public int getAlgProgressSts() {
        if (videoRender != null) {
            return videoRender.getAlgProgressSts();
        }
        return 0;
    }
    public int setRenderBitmap(Bitmap bitmap) {
        if (videoRender != null) {
            return videoRender.setRenderBitmap(bitmap);
        } else {
            return -1;
        }
    }

    public void setSplitScreenPos(float posRatio) {
        if (videoRender != null) {
            videoRender.setSplitScreenPos(posRatio);
        }
    }

    public void setAlgInputParam(String stringParam) {
        if (videoRender != null) {
            videoRender.setAlgInputParam(stringParam);
        }
    }

    public void setAlgDependParams(String[] dependParams) {
        if (videoRender != null) {
            videoRender.setAlgDependParams(dependParams);
        }
    }

    public void setSplitScreenMode(int splitScreenMode) {
        if (videoRender != null) {
            videoRender.setSplitScreenMode(splitScreenMode);
        }
    }
    public GLSurfaceViewRender getVideoRender() { return videoRender; }
    public void closeRender() {
        if (videoRender != null) {
            videoRender.close();
        }
    }
}
