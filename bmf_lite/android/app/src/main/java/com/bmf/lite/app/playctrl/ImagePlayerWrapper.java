package com.bmf.lite.app.playctrl;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.media.ExifInterface;
import android.media.MediaPlayer;
import android.opengl.GLES20;
import android.util.Log;
import android.view.Surface;

import com.bmf.lite.app.render.GLSurfaceViewWrapper;
import com.bmf.lite.app.render.TextureUtils;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;

public class ImagePlayerWrapper {
    private boolean loopPlaySts = true;
    private GLSurfaceViewWrapper glSurfaceView = null;
    private String videoPath =
        ""; /// storage/emulated/0/DCIM/Camera/VID_20230307_104326.mp4";
    private String TAG = "bmf-demo-app MediaPlayerWrapper";
    Bitmap bitmap = null;
    public ImagePlayerWrapper(GLSurfaceViewWrapper surfaceView) {
        glSurfaceView = surfaceView;
        glSurfaceView.setOesStatus(false);
        glSurfaceView.initRender();
    }

    public void setOesStatus(boolean isOes) {
        glSurfaceView.setOesStatus(isOes);
    }

    public void loadBitmap(String imgPath) {
        BitmapFactory.Options bitmapOptions = new BitmapFactory.Options();
        //        bitmapOptions.inSampleSize = 8;
        int degree = computeImgDegree(new File(imgPath).getAbsolutePath());
        Bitmap cameraBitmap = BitmapFactory.decodeFile(imgPath, bitmapOptions);
        bitmap = rotateImage(degree, cameraBitmap);
    }

    public static Bitmap rotateImage(int angle, Bitmap bitmap) {
        Matrix matrix = new Matrix();
        matrix.postRotate(angle);
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        Bitmap resultBitmap =
            Bitmap.createBitmap(bitmap, 0, 0, width, height, matrix, true);
        return resultBitmap;
    }

    public static int computeImgDegree(String dir) {
        int result = 0;
        try {
            ExifInterface exifInteface = new ExifInterface(dir);
            int orientation =
                exifInteface.getAttributeInt(ExifInterface.TAG_ORIENTATION,
                                             ExifInterface.ORIENTATION_NORMAL);
            switch (orientation) {
            case ExifInterface.ORIENTATION_ROTATE_270:
                result = 270;
                break;
            case ExifInterface.ORIENTATION_ROTATE_180:
                result = 180;
                break;
            case ExifInterface.ORIENTATION_ROTATE_90:
                result = 90;
                break;
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return result;
    }

    public int setCurImgResource(String imgPath) throws IOException {
        int textureID = -1;
        try {
            while (true) {
                textureID = glSurfaceView.getVideoRender()
                                .getAlgRender()
                                .getTextureID();
                if (textureID > 0) {
                    break;
                }
                Thread.sleep(1);
            }
        } catch (Exception e) {
            e.printStackTrace();
            Log.e(TAG, "load !");
            return -1;
        }
        loadBitmap(imgPath);
        int res = -1;
        try {
            while (true) {
                res = glSurfaceView.setRenderBitmap(bitmap);
                if (res == 0) {
                    break;
                }
                Thread.sleep(1);
            }
        } catch (Exception e) {
            e.printStackTrace();
            Log.e(TAG, "Thread problem!");
            return -2;
        }
        glSurfaceView.setOesStatus(false);
        glSurfaceView.setSplitScreenPos(0.75f);
        glSurfaceView.setSplitScreenMode(2);
        return 0;
    }

    public void switchAlgorithm(int algType) {
        if (glSurfaceView != null) {
            glSurfaceView.switchAlgorithm(algType);
        }
    }

    public int getAlgProgressSts() {
        if (glSurfaceView != null) {
            return glSurfaceView.getAlgProgressSts();
        }
        return 0;
    }
    public void stop() {}

    public void resume() {}

    public void pause() {}

    public void setLooping(boolean isLoop) {}

    public void destroy() {}
}
