package com.bmf.lite.app.algorithm;

import android.graphics.SurfaceTexture;
import android.opengl.GLES20;
import android.opengl.GLUtils;
import android.util.Log;

import com.bmf.lite.VideoFrameOutput;
import com.bmf.lite.app.render.OesTo2dTex;
import com.bmf.lite.app.render.SplitScreenRender;
import com.bmf.lite.app.render.TextureUtils;
import com.bmf.lite.AlgorithmInterface;
import com.bmf.lite.VideoFrame;
import com.bmf.lite.Param;
import android.graphics.Bitmap;

import java.security.SecureRandom;
import com.bmf.lite.common.ErrorCode;

public class AlgorithmRenderer {
    public enum ALG_TYPE {
        NORMAL,
        SUPERRESOLUTION,
        DENOISE,
        TEXGENIMG,
    }

    public enum RENDER_MODE {
        NORMAL,
        LEFT_RIGHT_SPLIT_SCREEN,
        UP_DOWN_SPLIT_SCREEN,
    }

    private String TAG = "bmf-demo-app AlgRenderer";
    private ALG_TYPE algType = ALG_TYPE.SUPERRESOLUTION;
    private int videoWidth = 720;
    private int videoHeight = 1280;
    private int wndWidth = -1;
    private int wndHeight = -1;
    private int lastVideoWidth = -1;
    private int lastVideoHeight = -1;
    private int lastWndWidth = -1;
    private int lastWndHeight = -1;
    private boolean initStatus = false;
    private OesTo2dTex oesTo2dTex = null;
    private SplitScreenRender screenRender = null;
    private boolean textureInitStatus = false;
    private int inputTextureId = -1;
    private int oesTo2DTextureId = -1;
    private int algOutputTextureId = -1; // demo create
    private int algReturnTextureId = -1; // alg create
    private SurfaceTexture surfaceTexture = null;
    boolean isOesTexture = true;
    int algTypeIndex = 0;
    AlgorithmInterface algorithmInterface = null;
    Param initParam = null;
    VideoFrame videoFrame = null;
    Param processParam = null;
    boolean resolutionChangeSts = false;
    boolean algInitStatus = false;
    boolean isDumpFrame = false;
    private final int DUMP_FRMAE_INDEX = 40;
    int frameIndex = 0;
    private int displayDividerStatus = 1;
    private float splitScreenRatio = 0.5f;
    private int renderStatus = -1;
    private RENDER_MODE renderOnScreenMode = RENDER_MODE.NORMAL;
    Bitmap bitmapFrame = null;
    Bitmap lastBitmapFrame = null;

    String algRenderParam = "";
    int algProgressSts = 0;

    String[] algDependParams = null;

    private float[] videoDecodeMatrix = {
        1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
    };
    public AlgorithmRenderer() {}
    public void setOesStatus(boolean isOes) {
        isOesTexture = isOes;
        renderStatus = 0;
    }
    public void switchAlgorithm(int algTypeId) {
        if (algTypeIndex != algTypeId) {
            closeAlg();
            // close old alg
            // open new alg
            algTypeIndex = algTypeId;
            algInitStatus = false;
            switch (algTypeIndex) {
            case 1:
                algType = ALG_TYPE.SUPERRESOLUTION;
                break;
            case 2:
                algType = ALG_TYPE.DENOISE;
                break;
            case 3:
                algType = ALG_TYPE.TEXGENIMG;
                break;
            default:
                break;
            }
        }
    }

    private int initAlg() {
        int ret = 0;
        if (algorithmInterface == null) {
            algorithmInterface = new AlgorithmInterface();
            if (algorithmInterface == null) {
                Log.d(TAG, "AlgorithmInterface new fail ");
                return ErrorCode.INSUFFICIENT_CPU_MEMORY;
            }
            ret = algorithmInterface.init();
            if (ret != 0) {
                Log.d(TAG, "algorithmInterface init ret = " + ret);
                return ret;
            }
        }
        if (initParam == null) {
            initParam = new Param();
            ret = initParam.init();
            if (ret != 0) {
                Log.d(TAG, "initParam init ret = " + ret);
                return ret;
            }
        }
        if (processParam == null) {
            processParam = new Param();
            ret = processParam.init();
            if (ret != 0) {
                Log.d(TAG, "processParam init ret = " + ret);
                return ret;
            }
        }
        if (videoFrame == null) {
            videoFrame = new VideoFrame();
            if (ret != 0) {
                Log.d(TAG, "videoFrame init ret = " + ret);
                return ret;
            }
        }
        initParam.setInt("change_mode", 6);
        initParam.setInt("algorithm_version", 0);
        switch (algTypeIndex) {
        case 1:
            initParam.setString("instance_id", "sr1");
            initParam.setInt("scale_mode", 0);
            initParam.setInt("algorithm_type", 0);
            initParam.setInt("sharp_levels", 0);
            initParam.setString("weight_path", "");
            break;
        case 2:
            initParam.setString("instance_id", "denoise");
            initParam.setInt("algorithm_type", 1);
            break;
        case 3:
            initParam.setString("instance_id", "tex2pic");
            initParam.setInt("algorithm_type", 3);
            if (algDependParams.length < 9) {
                Log.e(TAG,
                      "algorithmInterface init setInitParam fail, algorithm " +
                      "dependency parameters are not set correctly.");
                return ret;
            }
            initParam.setString("ld_library_path", algDependParams[0]);
            initParam.setString("adsp_system_library_path", algDependParams[1]);
            initParam.setString("qnn_htp_library_path", algDependParams[2]);
            initParam.setString("qnn_system_library_path", algDependParams[3]);
            initParam.setString("tokenizer_path", algDependParams[4]);
            initParam.setString("unet_path", algDependParams[5]);
            initParam.setString("text_encoder_path", algDependParams[6]);
            initParam.setString("vae_path", algDependParams[7]);
            initParam.setString("control_net_path", algDependParams[8]);
            algProgressSts = -1000;
            break;
        default:
            break;
        }
        initParam.setInt("backend", 3);
        initParam.setInt("process_mode", 0);
        initParam.setInt("max_width", 1920);
        initParam.setInt("max_height", 1080);
        initParam.setString("license_module_name", "");
        initParam.setString("program_cache_dir", "");
        ret = algorithmInterface.setParam(initParam);
        if (ret != 0) {
            Log.e(TAG, "algorithmInterface init setInitParam ret = " + ret +
                           ", algTypeIndex = " + algTypeIndex);
            return ret;
        }
        algInitStatus = true;
        return ret;
    }

    int algProcess() {
        int ret = 0;
        int inputTexId = inputTextureId;
        if (isOesTexture == true) {
            inputTexId = oesTo2DTextureId;
        }
        if (false == algInitStatus) {
            ret = initAlg();
            if (ret != 0) {
                Log.d(TAG, "algorithmInterface initAlg ret = " + ret);
                return ret;
            }
            if (algTypeIndex == 1) { // superresolution
                processParam.setInt("sharp_level", 0);
                processParam.setInt("scale_mode", 0);
            } else if (algTypeIndex == 3) {
                String positivePromptEn = "cute,girl";
                String negativePromptEn = "";
                processParam.setString("positive_prompt_en",
                                       positivePromptEn); // for tex2pic only
                processParam.setString("negative_prompt_en",
                                       negativePromptEn); // for tex2pic only
            }
            ret = videoFrame.init(inputTexId, videoWidth, videoHeight);
            if (ret != 0) {
                Log.d(TAG, "algorithmInterface videoFrame init ret = " + ret +
                               " w " + videoWidth + " h " + videoHeight);
                return ret;
            }
            resolutionChangeSts = false;
            algInitStatus = true;
        }
        if (resolutionChangeSts == true) {
            if (videoFrame != null) {
                videoFrame.Free();
            }
            ret = videoFrame.init(inputTexId, videoWidth, videoHeight);
            if (ret != 0) {
                Log.d(TAG, "algorithmInterface videoFrame init ret = " + ret +
                               " w " + videoWidth + " h " + videoHeight);
                return ret;
            }
            resolutionChangeSts = false;
        }

        if (isDumpFrame == true && frameIndex > DUMP_FRMAE_INDEX) {
            TextureUtils.SaveTexture2Bitmap(inputTexId, videoWidth, videoHeight,
                                            "/input_" + frameIndex + ".png",
                                            GLES20.GL_TEXTURE_2D);
        }
        if (algTypeIndex == 3) {
            if (algRenderParam == "reservedflagformarkgeneration") {
                processParam.setInt("new_prompt", 0); // for tex2pic only
                algProgressSts =
                    algProgressSts <= 100 ? algProgressSts + 25 : 100;
                if (algProgressSts >= 100) {
                    return 1; // generate ok
                }
            } else if (algRenderParam != "") {
                processParam.setString("positive_prompt_en",
                                       algRenderParam); // for tex2pic only
                processParam.setString("negative_prompt_en",
                                       ""); // for tex2pic onlySecureRandom
                SecureRandom secureRandom = new SecureRandom();
                int randomInt = secureRandom.nextInt(200) - 100;
                processParam.setInt("seed", randomInt); // for tex2pic only
                processParam.setInt("step", 5);         // for tex2pic only
                processParam.setInt("new_prompt", 1);   // for tex2pic only
                algRenderParam = "reservedflagformarkgeneration";
                algProgressSts = 0;
            }
        }
        ret = algorithmInterface.processVideoFrame(videoFrame, processParam);
        if (ret != 0) {
            Log.d(TAG, "algorithmInterface processVideoFrame ret = " + ret);
            return ret;
        }
        VideoFrameOutput videoFrameOutput =
            algorithmInterface.getVideoFrameOutput();
        if (videoFrameOutput.outputStatus != 0) {
            Log.d(TAG, "algorithmInterface getVideoFrameOutput ret = " +
                           videoFrameOutput.outputStatus + " width " +
                           videoFrameOutput.frame.getWidth() + " height " +
                           videoFrameOutput.frame.getHeight());
            return videoFrameOutput.outputStatus;
        }
        algReturnTextureId = videoFrameOutput.frame.getTextureId();
        if (isDumpFrame == true && frameIndex > DUMP_FRMAE_INDEX) {
            TextureUtils.SaveTexture2Bitmap(
                algReturnTextureId, videoFrameOutput.frame.getWidth(),
                videoFrameOutput.frame.getHeight(),
                "/output_" + frameIndex + ".png", GLES20.GL_TEXTURE_2D);
            isDumpFrame = false;
        }
        //        float psnr =
        //        TextureUtils.PSNR(inputTexId,algReturnTextureId,videoWidth,
        //        videoHeight,true);//denoise Logging.d("input + output
        //        psnr:"+psnr);
        frameIndex++;
        videoFrameOutput.Free();
        return 0;
    }

    private int closeAlg() {
        if (algorithmInterface != null) {
            algorithmInterface.Free();
            algorithmInterface = null;
        }
        if (initParam != null) {
            initParam.Free();
            initParam = null;
        }
        if (processParam != null) {
            processParam.Free();
            processParam = null;
        }
        if (videoFrame != null) {
            videoFrame.Free();
            videoFrame = null;
        }
        algInitStatus = false;
        return 0;
    }

    private void initInstance() {
        do {
            screenRender = new SplitScreenRender(); // CommonScreenRender();
            initStatus = screenRender.init();
            if (!initStatus) {
                Log.e(TAG, "Bmf-modules base shader init failed.");
                break;
            }
            oesTo2dTex = new OesTo2dTex();
            initStatus = oesTo2dTex.init();
            if (!initStatus) {
                Log.e(TAG, "Bmf-modules OesTo2d shader init failed.");
                break;
            }
            Log.i(TAG, "ScreenRender init success.");
            return;
        } while (false);
        screenRender = null;
        oesTo2dTex = null;
    }

    private void resetTexture() {
        if (oesTo2DTextureId != -1) {
            GLES20.glDeleteTextures(1, new int[] {oesTo2DTextureId}, 0);
        }
        if (algOutputTextureId != -1) {
            GLES20.glDeleteTextures(1, new int[] {algOutputTextureId}, 0);
        }
        int[] textureArr = TextureUtils.CreateTextures(2);
        oesTo2DTextureId = textureArr[0];
        algOutputTextureId = textureArr[1];
        textureInitStatus = false;
        initTexture();
    }
    private boolean onResolutionChanged() {
        boolean resChange = false;
        if (lastVideoWidth != videoWidth || lastVideoHeight != videoHeight) {
            lastVideoWidth = videoWidth;
            lastVideoHeight = videoHeight;
            resChange = true;
            resetTexture();
            resolutionChangeSts = true;
        }
        if (lastWndWidth != wndWidth || lastWndHeight != wndHeight) {
            lastWndWidth = wndWidth;
            lastWndHeight = wndHeight;
            resChange = true;
        }
        return resChange;
    }

    public void setRenderImgSize(int width, int height) {
        Log.e(TAG, "set videoWidth:" + width + " videoHeight:" + height);
        videoWidth = width;
        videoHeight = height;
    }

    public void setRenderWndSize(int width, int height) {
        wndWidth = width;
        wndHeight = height;
    }

    public void setTextureID(int textureID) {
        inputTextureId = textureID;
        surfaceTexture = new SurfaceTexture(textureID);
        initInstance();
        screenRender.setSplitScreenMode(displayDividerStatus);
        screenRender.setSplitScreenPos(splitScreenRatio);
    }

    public SurfaceTexture getSurfaceTexture() {
        if (surfaceTexture != null) {
            renderStatus = 0;
            return surfaceTexture;
        }
        return null;
    }

    public int getTextureID() { return inputTextureId; }

    public void updateTexture() {
        initTexture();
        if (surfaceTexture != null) {
            surfaceTexture.updateTexImage();
            surfaceTexture.getTransformMatrix(videoDecodeMatrix);
        } else {
            Log.d(TAG, "surfaceTexture is null");
        }
    }

    private void initTexture() {
        if (!initStatus) {
            initInstance();
        }
        if (!textureInitStatus) {
            Log.d(TAG, "init videoWidth:" + videoWidth +
                           " videoHeight:" + videoHeight);
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, oesTo2DTextureId);
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
            GLES20.glTexImage2D(GLES20.GL_TEXTURE_2D, 0, GLES20.GL_RGBA,
                                videoWidth, videoHeight, 0, GLES20.GL_RGBA,
                                GLES20.GL_UNSIGNED_BYTE, null);
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, 0);
            textureInitStatus = true;
        }
    }
    public int setRenderBitmap(Bitmap bitmap) {
        bitmapFrame = bitmap;
        return 0;
    }

    public void setAlgDependParams(String[] dependParams) {
        algDependParams = dependParams;
    }

    public int updateBitmapToTexture(int textId) {
        if (lastBitmapFrame == bitmapFrame) {
            return 0;
        }
        videoWidth = bitmapFrame.getWidth();
        videoHeight = bitmapFrame.getHeight();
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, textId);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D,
                               GLES20.GL_TEXTURE_MIN_FILTER,
                               GLES20.GL_LINEAR_MIPMAP_LINEAR);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D,
                               GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR);

        GLUtils.texImage2D(GLES20.GL_TEXTURE_2D, 0, bitmapFrame, 0);
        GLES20.glGenerateMipmap(GLES20.GL_TEXTURE_2D);

        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, 0);
        lastBitmapFrame = bitmapFrame;
        if (!bitmapFrame.isRecycled()) {
            bitmapFrame.recycle();
        }
        TextureUtils.LogOESError("TextureUtils");
        return 0;
    }

    public void draw() {
        if (renderStatus < 0) {
            return;
        }
        if (renderOnScreenMode == RENDER_MODE.UP_DOWN_SPLIT_SCREEN) {
            updateBitmapToTexture(inputTextureId);
        } else {
            updateTexture();
        }
        if (onResolutionChanged() == true) {
            Log.d(TAG, "updatePrjMatrix videoWidth:" + videoWidth +
                           " videoHeight:" + videoHeight + " wndWidth:" +
                           wndWidth + " wndHeight:" + wndHeight);
            screenRender.updatePrjMatrix(videoWidth, videoHeight, wndWidth,
                                         wndHeight);
        }
        int ret = 0;
        if (isOesTexture == true) {
            oesTo2dTex.setMatrix(videoDecodeMatrix);
            oesTo2dTex.process(inputTextureId, oesTo2DTextureId, videoWidth,
                               videoHeight);
            switch (algType) {
            case SUPERRESOLUTION:
            case DENOISE:
                ret = algProcess();
                Log.d(TAG, "algProcess,type = " + algType +
                               ",input-oesTexId = " + inputTextureId +
                               " 2dTexId = " + oesTo2DTextureId +
                               " algOutputTex = " + algReturnTextureId +
                               " ret = " + ret);
                screenRender.drawToScreen(oesTo2DTextureId, algReturnTextureId,
                                          wndWidth, wndHeight);
                break;
            case TEXGENIMG:
            default:
                screenRender.drawToScreen(oesTo2DTextureId, oesTo2DTextureId,
                                          wndWidth, wndHeight);
                Log.d(TAG, "draw oes normal ");
                break;
            }
        } else {
            switch (algType) {
            case SUPERRESOLUTION:
            case DENOISE:
                break;
            case TEXGENIMG:
                if (algRenderParam != "") {
                    ret = algProcess();
                    if (algInitStatus == false) {
                        algProgressSts = -200; // 不支持
                    }
                    if (ret != 0) {
                        algRenderParam = "";
                    }
                    Log.d(TAG, "algProcess,type = " + algType +
                                   ",input-texId = " + inputTextureId +
                                   " 2dTexId = " + oesTo2DTextureId +
                                   " algOutputTex = " + algReturnTextureId +
                                   " ret = " + ret);
                }
                screenRender.drawToScreen(inputTextureId, algReturnTextureId,
                                          wndWidth, wndHeight);
                break;
            default:
                screenRender.drawToScreen(inputTextureId, inputTextureId,
                                          wndWidth, wndHeight);
                Log.d(TAG, "draw 2d texture normal ");
                break;
            }
        }
    }

    @Override
    protected void finalize() {
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, 0);
        if (oesTo2DTextureId != -1) {
            GLES20.glDeleteTextures(1, new int[] {oesTo2DTextureId}, 0);
            oesTo2DTextureId = -1;
        }
        if (algOutputTextureId != -1) {
            GLES20.glDeleteTextures(1, new int[] {algOutputTextureId}, 0);
            algOutputTextureId = -1;
        }
        if (inputTextureId != -1) {
            GLES20.glDeleteTextures(1, new int[] {inputTextureId}, 0);
            inputTextureId = -1;
        }
        if (surfaceTexture != null) {
            surfaceTexture.release();
            surfaceTexture = null;
        }
        renderStatus = -1;
        closeAlg();
    }

    public void setAlgInputAndOutputTexture(int inputTexture,
                                            int outputTexture) {
        oesTo2DTextureId = inputTexture;
        algOutputTextureId = outputTexture;
    }

    public void setSplitScreenPos(float posRatio) {
        if (screenRender != null) {
            screenRender.setSplitScreenPos(posRatio);
        } else {
            splitScreenRatio = posRatio;
        }
    }

    public void setAlgInputParam(String stringParam) {
        if (algRenderParam != stringParam && stringParam != "") {
            algRenderParam = stringParam;
            algProgressSts = 0;
        }
    }
    public void setSplitScreenMode(int splitScreenMode) {
        if (splitScreenMode == 0) {
            renderOnScreenMode = RENDER_MODE.NORMAL;
        } else if (splitScreenMode == 1) {
            renderOnScreenMode = RENDER_MODE.LEFT_RIGHT_SPLIT_SCREEN;
        } else if (splitScreenMode == 2) {
            renderOnScreenMode = RENDER_MODE.UP_DOWN_SPLIT_SCREEN;
        }
        if (screenRender != null) {
            screenRender.setSplitScreenMode(splitScreenMode);
        } else {
            displayDividerStatus = splitScreenMode;
        }
    }

    public int getAlgProgressSts() { return algProgressSts; }
}
