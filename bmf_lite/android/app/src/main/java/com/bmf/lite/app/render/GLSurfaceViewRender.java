package com.bmf.lite.app.render;

import android.opengl.GLES20;
import android.opengl.GLSurfaceView;
import android.util.Log;

import androidx.annotation.Nullable;

import com.bmf.lite.app.algorithm.AlgorithmRenderer;
import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;
import android.graphics.Bitmap;

public class GLSurfaceViewRender implements GLSurfaceView.Renderer {
    private AlgorithmRenderer algRenderer = new AlgorithmRenderer();
    private String TAG = "bmf-demo-app GLSurfaceViewRender";
    public GLSurfaceViewRender() {}
    public void close() {}
    @Override
    public void onSurfaceCreated(GL10 gl, EGLConfig config) {
        GLES20.glClearColor(0f, 0f, 0f, 0f);
        GLES20.glEnable(GLES20.GL_BLEND);
        GLES20.glBlendFunc(GLES20.GL_SRC_ALPHA, GLES20.GL_ONE_MINUS_SRC_ALPHA);
        int[] textureIds = new int[3];
        GLES20.glGenTextures(3, textureIds, 0);
        algRenderer.setTextureID(textureIds[0]);
        algRenderer.setAlgInputAndOutputTexture(textureIds[1], textureIds[2]);
    }
    @Override
    public void onSurfaceChanged(GL10 gl, int width, int height) {
        GLES20.glViewport(0, 0, width, height);
        Log.d(TAG, "current mSurfaceViewWidth onSurfaceChanged:" + width +
                       " height:" + height);
        algRenderer.setRenderWndSize(width, height);
    }

    public void setRenderImgSize(int width, int height) {
        algRenderer.setRenderImgSize(width, height);
    }

    @Override
    public boolean equals(@Nullable Object obj) {
        return super.equals(obj);
    }

    public void setRenderWndSize(int width, int height) {
        algRenderer.setRenderWndSize(width, height);
    }
    @Override
    public void onDrawFrame(GL10 gl) {
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT | GLES20.GL_DEPTH_BUFFER_BIT);
        algRenderer.draw();
    }

    public void setOesStatus(boolean isOes) { algRenderer.setOesStatus(isOes); }

    public void switchAlgorithm(int algType) {
        if (algRenderer != null) {
            algRenderer.switchAlgorithm(algType);
        }
    }

    public int setRenderBitmap(Bitmap bitmap) {
        if (algRenderer != null) {

            return algRenderer.setRenderBitmap(bitmap);
        }
        return -1;
    }

    public void setAlgDependParams(String[] dependParams) {
        if (algRenderer != null) {
            algRenderer.setAlgDependParams(dependParams);
        }
    }

    public AlgorithmRenderer getAlgRender() { return algRenderer; }

    public void setSplitScreenPos(float posRatio) {
        if (algRenderer != null) {
            algRenderer.setSplitScreenPos(posRatio);
        }
    }
    public void setAlgInputParam(String stringParam) {
        if (algRenderer != null) {
            algRenderer.setAlgInputParam(stringParam);
        }
    }
    public void setSplitScreenMode(int splitScreenMode) {
        if (algRenderer != null) {
            algRenderer.setSplitScreenMode(splitScreenMode);
        }
    }
    public int getAlgProgressSts() {
        if (algRenderer != null) {
            return algRenderer.getAlgProgressSts();
        }
        return 0;
    }
}