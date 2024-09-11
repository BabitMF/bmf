package com.bmf.lite.app.render;

import android.opengl.GLES20;
import android.opengl.Matrix;
import android.util.Log;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

public class SplitScreenRender {
    private String TAG = "bmf-demo-app SplitScreenRender";
    protected boolean initStatus = false;
    protected int shaderProgram = 0;
    protected int vertexShader;
    protected int fragmentShader;
    protected int positiveHandler;
    protected int coordHandler;
    protected FloatBuffer vertextBuffer;
    protected FloatBuffer texcoordBuffer;
    private int frameTextureLeftHandler = -1;
    private int frameTextureRightHandler = -1;
    private int splitScreenLRRatioHandler = 0;
    private int screenRenderModeHandler = 0;
    private int screenRenderMode = 1;
    private float splitScreenRatio = 0.5f;
    protected static float[] POSITION_VERTEX = {
        -1.0f, -1.0f, 0.0f, 1.0f, -1f,  0.0f,
        -1.0f, 1.0f,  0.0f, 1.0f, 1.0f, 0.0f,
    };

    protected static float[] TEXTURE_VERTEX = {
        0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f,
    };
    private String vertexShaderCode = "attribute vec4 vPosition;"
                                      + "uniform mat4 uMvpMatrix;"
                                      + "attribute vec4 vCoord;"
                                      + "varying vec2 aCoord;"
                                      + "void main() {"
                                      + "  gl_Position = uMvpMatrix*vPosition;"
                                      + "  aCoord = vCoord.xy;"
                                      + "}";
    private String fragmentShaderCode =
        "#extension GL_OES_EGL_image_external : require\n"
        + "precision mediump float;"
        + "uniform sampler2D leftTexture;"
        + "uniform sampler2D rightTexture;"
        + "uniform float splitRatio;"
        + "uniform int screenRenderMode;"
        + "varying vec2 aCoord;"
        + "void main() {"
        + "  float divWidth = 0.003;"
        + "  vec2 texCoord = aCoord;"
        + "  if(screenRenderMode != 1)"
        + "  {"
        + "    divWidth = 0.0;"
        + "  }"
        + "  if(screenRenderMode == 2)"
        + "  {"
        + "    texCoord.y = 1.0-texCoord.y;"
        + "  }"
        + "  if(aCoord.x<=splitRatio-divWidth)"
        + "  {"
        + "    gl_FragColor = texture2D(leftTexture, texCoord);"
        + "  }"
        + "  else if(texCoord.x>splitRatio-divWidth && "
        + "texCoord.x<=splitRatio+divWidth && screenRenderMode>0)"
        + "  {"
        + "    gl_FragColor = vec4(1.0,0.0,0.0,1.0);"
        + "  }"
        + "  else"
        + "  {"
        + "    gl_FragColor = texture2D(rightTexture, texCoord);"
        + "  }"
        + "}";
    private int mvpMatrixHandle;
    private float[] mvpMatrix = {
        1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
    };
    private float[] mvpExtMatrix = {
        1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
    };

    private float[] projectMatrix = {
        1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
    };

    private float[] viewMatrix = {
        1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
    };

    public SplitScreenRender() {}

    public Boolean init() {
        if (initStatus) {
            return initStatus;
        }
        createMProgram();
        return initStatus;
    }
    public static int loadShader(int type, String shaderCode) {
        int shader = GLES20.glCreateShader(type);
        GLES20.glShaderSource(shader, shaderCode);
        GLES20.glCompileShader(shader);
        return shader;
    }

    public void createMProgram() {
        ByteBuffer bb = ByteBuffer.allocateDirect(POSITION_VERTEX.length * 4);
        vertextBuffer = bb.order(ByteOrder.nativeOrder()).asFloatBuffer();
        vertextBuffer.put(POSITION_VERTEX);
        vertextBuffer.position(0);
        ByteBuffer m_bb = ByteBuffer.allocateDirect(TEXTURE_VERTEX.length * 4);
        texcoordBuffer = m_bb.order(ByteOrder.nativeOrder()).asFloatBuffer();
        texcoordBuffer.put(TEXTURE_VERTEX);
        texcoordBuffer.position(0);
        vertexShader = loadShader(GLES20.GL_VERTEX_SHADER, vertexShaderCode);
        fragmentShader =
            loadShader(GLES20.GL_FRAGMENT_SHADER, fragmentShaderCode);
        shaderProgram = GLES20.glCreateProgram();
        GLES20.glAttachShader(shaderProgram, vertexShader);
        GLES20.glAttachShader(shaderProgram, fragmentShader);
        GLES20.glLinkProgram(shaderProgram);
        GLES20.glDeleteShader(vertexShader);
        GLES20.glDeleteShader(fragmentShader);
        positiveHandler =
            GLES20.glGetAttribLocation(shaderProgram, "vPosition");
        coordHandler = GLES20.glGetAttribLocation(shaderProgram, "vCoord");
        mvpMatrixHandle =
            GLES20.glGetUniformLocation(shaderProgram, "uMvpMatrix");
        frameTextureLeftHandler =
            GLES20.glGetUniformLocation(shaderProgram, "leftTexture");
        frameTextureRightHandler =
            GLES20.glGetUniformLocation(shaderProgram, "rightTexture");
        splitScreenLRRatioHandler =
            GLES20.glGetUniformLocation(shaderProgram, "splitRatio");
        screenRenderModeHandler =
            GLES20.glGetUniformLocation(shaderProgram, "screenRenderMode");
        if (GLES20.glGetError() == GLES20.GL_NO_ERROR) {
            initStatus = true;
        }
    }

    public synchronized void drawToScreen(int texture_id_1, int texture_id_2,
                                          int world_width, int world_height) {
        GLES20.glUseProgram(shaderProgram);
        //        Log.d("","splitScreenRatio "+splitScreenRatio+"
        //        screenRenderMode "+screenRenderMode);
        if (screenRenderMode == 2) {
            GLES20.glViewport(0, (int)(world_height * splitScreenRatio),
                              world_width,
                              (int)(world_height * (1 - splitScreenRatio)));
        } else {
            GLES20.glViewport(0, 0, world_width, world_height);
        }
        GLES20.glVertexAttribPointer(positiveHandler, 3, GLES20.GL_FLOAT, false,
                                     12, vertextBuffer);
        GLES20.glEnableVertexAttribArray(positiveHandler);
        GLES20.glVertexAttribPointer(coordHandler, 2, GLES20.GL_FLOAT, false, 8,
                                     texcoordBuffer);
        GLES20.glEnableVertexAttribArray(coordHandler);
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, texture_id_1);
        if (screenRenderMode == 1) {
            GLES20.glActiveTexture(GLES20.GL_TEXTURE1);
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, texture_id_2);
        }
        GLES20.glUniform1i(frameTextureLeftHandler, 0);
        GLES20.glUniform1i(frameTextureRightHandler, 1);

        if (screenRenderMode == 2) {
            GLES20.glUniform1f(splitScreenLRRatioHandler, 1.0f);
        } else {
            GLES20.glUniform1f(splitScreenLRRatioHandler, splitScreenRatio);
        }
        GLES20.glUniform1i(screenRenderModeHandler, screenRenderMode);
        GLES20.glUniformMatrix4fv(mvpMatrixHandle, 1, false, mvpMatrix, 0);
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, 0);
        GLES20.glDisableVertexAttribArray(positiveHandler);
        GLES20.glDisableVertexAttribArray(coordHandler);

        if (screenRenderMode == 2) {
            GLES20.glUseProgram(shaderProgram);
            GLES20.glViewport(0, 0, world_width,
                              (int)(world_height * splitScreenRatio));
            GLES20.glVertexAttribPointer(positiveHandler, 3, GLES20.GL_FLOAT,
                                         false, 12, vertextBuffer);
            GLES20.glEnableVertexAttribArray(positiveHandler);
            GLES20.glVertexAttribPointer(coordHandler, 2, GLES20.GL_FLOAT,
                                         false, 8, texcoordBuffer);
            GLES20.glEnableVertexAttribArray(coordHandler);
            GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, texture_id_2);
            GLES20.glUniform1i(frameTextureLeftHandler, 0);
            GLES20.glUniform1i(frameTextureRightHandler, 1);
            GLES20.glUniform1f(splitScreenLRRatioHandler, 1.0f);
            GLES20.glUniform1i(screenRenderModeHandler, screenRenderMode);
            GLES20.glUniformMatrix4fv(mvpMatrixHandle, 1, false, mvpExtMatrix,
                                      0);
            GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, 0);
            GLES20.glDisableVertexAttribArray(positiveHandler);
            GLES20.glDisableVertexAttribArray(coordHandler);
        }
    }

    @Override
    protected void finalize() {
        if (0 != shaderProgram) {
            GLES20.glDeleteProgram(shaderProgram);
        }
    }

    public float[] updatePrjMatrix(int imgWidth, int imgHeight, int wndWidth,
                                   int wndHeight) {
        int initWndHeight = wndHeight;
        if (screenRenderMode == 2) {
            //            wndHeight =  (int)(wndHeight*splitScreenRatio);
            wndHeight = (int)(initWndHeight * (1 - splitScreenRatio));
        }
        float wndRatio = 1.0f;
        float originRatio = 1.0f;
        float widthRatio = 1.0f;
        float heightRatio = 1.0f;
        if (imgWidth != -1 && imgHeight != -1 && wndWidth != -1 &&
            wndHeight != -1) {
            originRatio = imgWidth / (float)imgHeight;
            wndRatio = wndWidth / (float)wndHeight;
            if (wndWidth > wndHeight) {
                if (originRatio > wndRatio) {
                    heightRatio = originRatio / wndRatio;
                    Matrix.orthoM(projectMatrix, 0, -widthRatio, widthRatio,
                                  -heightRatio, heightRatio, 3f, 5f);
                } else {
                    widthRatio = wndRatio / originRatio;
                    Matrix.orthoM(projectMatrix, 0, -widthRatio, widthRatio,
                                  -heightRatio, heightRatio, 3f, 5f);
                }
            } else {
                if (originRatio > wndRatio) {
                    heightRatio = originRatio / wndRatio;
                    Matrix.orthoM(projectMatrix, 0, -widthRatio, widthRatio,
                                  -heightRatio, heightRatio, 3f, 5f);
                } else {
                    widthRatio = wndRatio / originRatio;
                    Matrix.orthoM(projectMatrix, 0, -widthRatio, widthRatio,
                                  -heightRatio, heightRatio, 3f, 5f);
                }
            }
            Matrix.setLookAtM(viewMatrix, 0, 0f, 0f, 5.0f, 0f, 0f, 0f, 0f, 1.0f,
                              0f);
            Matrix.multiplyMM(mvpMatrix, 0, projectMatrix, 0, viewMatrix, 0);
        }
        if (screenRenderMode == 2) {
            updatePrjMatrixExt(imgWidth, imgHeight, wndWidth,
                               (int)(initWndHeight * (splitScreenRatio)));
        }
        return mvpMatrix;
    }

    public float[] updatePrjMatrixExt(int imgWidth, int imgHeight, int wndWidth,
                                      int wndHeight) {
        float wndRatio = 1.0f;
        float originRatio = 1.0f;
        float widthRatio = 1.0f;
        float heightRatio = 1.0f;
        if (imgWidth != -1 && imgHeight != -1 && wndWidth != -1 &&
            wndHeight != -1) {
            originRatio = imgWidth / (float)imgHeight;
            wndRatio = wndWidth / (float)wndHeight;
            if (wndWidth > wndHeight) {
                if (originRatio > wndRatio) {
                    heightRatio = originRatio / wndRatio;
                    Matrix.orthoM(projectMatrix, 0, -widthRatio, widthRatio,
                                  -heightRatio, heightRatio, 3f, 5f);
                } else {
                    widthRatio = wndRatio / originRatio;
                    Matrix.orthoM(projectMatrix, 0, -widthRatio, widthRatio,
                                  -heightRatio, heightRatio, 3f, 5f);
                }
            } else {
                if (originRatio > wndRatio) {
                    heightRatio = originRatio / wndRatio;
                    Matrix.orthoM(projectMatrix, 0, -widthRatio, widthRatio,
                                  -heightRatio, heightRatio, 3f, 5f);
                } else {
                    widthRatio = wndRatio / originRatio;
                    Matrix.orthoM(projectMatrix, 0, -widthRatio, widthRatio,
                                  -heightRatio, heightRatio, 3f, 5f);
                }
            }
            Matrix.setLookAtM(viewMatrix, 0, 0f, 0f, 5.0f, 0f, 0f, 0f, 0f, 1.0f,
                              0f);
            Matrix.multiplyMM(mvpExtMatrix, 0, projectMatrix, 0, viewMatrix, 0);
        }
        return mvpExtMatrix;
    }

    public void setSplitScreenPos(float posRatio) {
        Log.d("", "setSplitScreenPos posRatio " + posRatio +
                      " screenRenderMode " + screenRenderMode);
        if (posRatio > -0.000001f && posRatio < 1.000001f) {
            splitScreenRatio = posRatio;
        }
    }

    public void setSplitScreenMode(int splitScreenMode) {
        screenRenderMode = splitScreenMode;
    }
}
