package com.bmf.lite.app.render;

import android.opengl.GLES20;
import android.util.Log;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.opengl.GLES11Ext;
import android.opengl.GLES20;
import android.opengl.GLUtils;
import android.util.Log;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import com.bmf.lite.app.tool.SingleApplication;
public class TextureUtils {
    private static String TAG = "bmf-demo-app TextureUtils";
    private static String TARGET_PATH = "/sdcard/bmf_test_data/benchmark/";

    private static int lastWidth = 0;
    private static int lastHeight = 0;
    private static byte[] inputTexColorBytes = null;
    private static byte[] outputTexColorBytes = null;
    private static ByteBuffer inputTexColorBuf = null;
    private static ByteBuffer outputTexColorBuf = null;
    private static int[] textureIdList = null;

    private static int[] textureDatas = null;
    public static int[] GenTexture(int n, int start) {
        int texture_id[] = new int[n];
        GLES20.glGenTextures(n, texture_id, start);
        for (int i = 0; i < n; ++i) {
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, texture_id[i]);
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D,
                                   GLES20.GL_TEXTURE_WRAP_S,
                                   GLES20.GL_CLAMP_TO_EDGE);
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D,
                                   GLES20.GL_TEXTURE_WRAP_T,
                                   GLES20.GL_CLAMP_TO_EDGE);
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D,
                                   GLES20.GL_TEXTURE_MIN_FILTER,
                                   GLES20.GL_LINEAR);
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D,
                                   GLES20.GL_TEXTURE_MAG_FILTER,
                                   GLES20.GL_LINEAR);
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, 0);
        }
        return texture_id;
    }

    public static int[] CreateTextures(int n) {
        int[] textures = new int[n];
        GLES20.glGenTextures(n, textures, 0);
        return textures;
    }

    public static void SaveBitmap(String name, Bitmap bitmap) {

        Log.d(TAG, "SaveBitmap Path = " + TARGET_PATH);
        File saveFile = new File(TARGET_PATH, name);
        if (saveFile.exists()) {
            saveFile.delete();
        }
        try {
            saveFile.createNewFile();
        } catch (IOException e) {
            e.printStackTrace();
        }
        try {
            FileOutputStream saveImgOut = new FileOutputStream(saveFile);
            bitmap.compress(Bitmap.CompressFormat.PNG, 80, saveImgOut);
            saveImgOut.flush();
            saveImgOut.close();
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }
    public static void SaveTexture2Bitmap(int texId, int width, int height,
                                          String path, int texture_type) {
        int sizeNeeded = width * height * 4;
        ByteBuffer buf = ByteBuffer.allocateDirect(sizeNeeded);
        buf.rewind();
        buf.order(ByteOrder.LITTLE_ENDIAN);
        // GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, texId);
        int[] frameBuffers = new int[1];
        GLES20.glGenFramebuffers(1, frameBuffers, 0);
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, frameBuffers[0]);
        GLES20.glFramebufferTexture2D(GLES20.GL_FRAMEBUFFER,
                                      GLES20.GL_COLOR_ATTACHMENT0, texture_type,
                                      texId, 0);
        int val = GLES20.glCheckFramebufferStatus(GLES20.GL_FRAMEBUFFER);
        if (val != GLES20.GL_FRAMEBUFFER_COMPLETE) {
            Log.e(TAG,
                  "SaveTexture2Bitmap glBindFramebuffer failed,error code:" +
                      val);
        }
        GLES20.glReadPixels(0, 0, width, height, GLES20.GL_RGBA,
                            GLES20.GL_UNSIGNED_BYTE, buf);

        // flip the image
        Bitmap bmp =
            Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        buf.rewind();
        bmp.copyPixelsFromBuffer(buf);
        if (path != "") {
            SaveBitmap(path, bmp);
        }
        bmp.recycle();

        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0);
        GLES20.glDeleteFramebuffers(1, new int[] {frameBuffers[0]}, 0);
    }

    public static void LogOESError(String moduleTAG) {
        int error = GLES20.glGetError();
        switch (error) {
        case GLES20.GL_NO_ERROR:
            Log.d(moduleTAG, "GL_OK");
            break;
        case GLES20.GL_INVALID_ENUM:
            Log.e(moduleTAG, "GL_INVALID_ENUM");
            break;
        case GLES20.GL_INVALID_VALUE:
            Log.e(moduleTAG, "GL_INVALID_VALUE");
            break;
        case GLES20.GL_INVALID_OPERATION:
            Log.e(moduleTAG, "GL_INVALID_OPERATION");
            break;
        case GLES20.GL_INVALID_FRAMEBUFFER_OPERATION:
            Log.e(moduleTAG, "GL_INVALID_FRAMEBUFFER_OPERATION");
            break;
        case GLES20.GL_OUT_OF_MEMORY:
            Log.e(moduleTAG, "GL_OUT_OF_MEMORY");
            break;
        default:
            Log.e(moduleTAG, "error happen" + String.valueOf(error));
            break;
        }
    }

    public static void CopyFromTexture(int texture, int width, int height,
                                       Buffer data) {
        if (textureIdList == null) {
            textureIdList = new int[1];
            GLES20.glGenFramebuffers(1, textureIdList, 0);
        }
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, textureIdList[0]);
        GLES20.glFramebufferTexture2D(GLES20.GL_FRAMEBUFFER,
                                      GLES20.GL_COLOR_ATTACHMENT0,
                                      GLES20.GL_TEXTURE_2D, texture, 0);
        int val = GLES20.glCheckFramebufferStatus(GLES20.GL_FRAMEBUFFER);
        if (val != GLES20.GL_FRAMEBUFFER_COMPLETE) {
            Log.e("TAG", "glBindFramebuffer failed,error code:" + val);
        }
        GLES20.glReadPixels(0, 0, width, height, GLES20.GL_RGBA,
                            GLES20.GL_UNSIGNED_BYTE, data);
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0);
    }
    public static float PSNR(int inputTextureId, int outputTextureId, int width,
                             int height, boolean isSameSize) {
        if (isSameSize == true) {
            int sizeNeeded = width * height * 4;
            if (lastWidth != width || lastHeight != height) {
                inputTexColorBytes = new byte[sizeNeeded];
                outputTexColorBytes = new byte[sizeNeeded];
                inputTexColorBuf = ByteBuffer.allocateDirect(sizeNeeded);
                outputTexColorBuf = ByteBuffer.allocateDirect(sizeNeeded);
                lastWidth = width;
                lastHeight = height;
            }
            inputTexColorBuf.rewind();
            inputTexColorBuf.order(ByteOrder.LITTLE_ENDIAN);
            CopyFromTexture(inputTextureId, width, height, inputTexColorBuf);
            inputTexColorBuf.rewind();
            inputTexColorBuf.get(inputTexColorBytes);
            outputTexColorBuf.rewind();
            outputTexColorBuf.order(ByteOrder.LITTLE_ENDIAN);
            CopyFromTexture(outputTextureId, width, height, outputTexColorBuf);
            outputTexColorBuf.rewind();
            outputTexColorBuf.get(outputTexColorBytes);
            if (inputTexColorBuf.hasRemaining()) {
                inputTexColorBuf.compact();
            } else {
                inputTexColorBuf.clear();
            }
            if (outputTexColorBuf.hasRemaining()) {
                outputTexColorBuf.compact();
            } else {
                outputTexColorBuf.clear();
            }
            //        long endTime4 = System.currentTimeMillis();
            //        Log.d("TextureUtils","PSNR copy data time:" + (endTime4 -
            //        startTime4) + " ms.");
            long sum = 0;
            int diff;
            for (int i = 0; i < sizeNeeded; i++) {
                int ioriginal = inputTexColorBytes[i] & 0xFF;
                int iresult = outputTexColorBytes[i] & 0xFF;
                diff = ioriginal - iresult;
                sum += diff * diff;
            }
            double MSE = sum / (1.0f * sizeNeeded);
            float PSNR = 0;
            if (MSE < 10e-10) {
                PSNR = 1000;
            } else {
                PSNR = (float)(10.0 * Math.log10((255.0 * 255.0) / MSE));
            }
            //        Log.d("TextureUtils ", "Sum : " +sum+" PSNR : " +PSNR);
            return PSNR;
        } else {
            Log.e(TAG, "psnr calculation with different resolutions is not "
                           + "supported.");
        }
        return -1000;
    }

    public static Bitmap decodeBitmapFromFile(File file) {
        InputStream inputStream = null;
        try {
            inputStream = new FileInputStream(file);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        Bitmap bitmap = BitmapFactory.decodeStream(inputStream);

        if (bitmap == null) {
            Log.e("TextureUtils", " could not be decoded.");
        }
        try {
            inputStream.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        return bitmap;
    }

    public static int[] loadBitmapToTexture(int texture_id, String path) {
        if (texture_id == 0) {
            Log.e("TextureUtils", " texture not init");
            return new int[] {texture_id, -1, -1};
        }

        int texture_type = GLES20.GL_TEXTURE_2D;
        Bitmap bitmap = decodeBitmapFromFile(new File(path));

        if (bitmap == null) {
            Log.e("TextureUtils", path + " could not be decoded.");
            return new int[] {texture_id, -1, -1};
        }
        if (textureDatas == null) {
            textureDatas = new int[3];
        }
        textureDatas[0] = texture_id;
        textureDatas[1] = bitmap.getWidth();
        textureDatas[2] = bitmap.getHeight();
        GLES20.glBindTexture(texture_type, texture_id);
        GLUtils.texImage2D(texture_type, 0, bitmap, 0);
        bitmap.recycle();
        GLES20.glBindTexture(texture_type, 0);
        TextureUtils.LogOESError("TextureUtils");
        return textureDatas;
    }

    public static Bitmap loadBitmapFromFile(String path) {
        Bitmap bitmap = decodeBitmapFromFile(new File(path));
        return bitmap;
    }
}
