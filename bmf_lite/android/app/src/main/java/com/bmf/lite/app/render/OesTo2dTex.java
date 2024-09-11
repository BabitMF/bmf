package com.bmf.lite.app.render;

import android.opengl.GLES11Ext;
import android.opengl.GLES20;
import android.util.Log;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

public class OesTo2dTex {
    private final String TAG = "bmf-demo-app OesTo2dTex";
    protected boolean initStatus = false;
    protected int glShaderProgram = -1;
    protected int vertexShader;
    protected int fragmentShader;
    protected int positiveHandler;
    protected int coordHandler;
    protected FloatBuffer vertextBuffer;
    protected FloatBuffer texcoordBuffer;
    private int frameTexture = -1;
    private int frameBuffer = -1;
    protected static float[] POSITION_VERTEX = {
        -1.0f, -1.0f, 1.0f, -1f, 1.0f, 1.0f, -1.0f, 1.0f,
    };
    protected static float[] TEXTURE_VERTEX = {
        -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f,
    };
    private final String vertexShaderCode =
        "uniform mat4 uSTMatrix;\n"
        + "attribute vec4 vPosition;\n"
        + "attribute vec4 vCoord;\n"
        + "varying vec2 aCoord;\n"
        + "void main() {\n"
        + "  gl_Position = vPosition;\n"
        + "  vec2 coord = (vCoord.xy /2.0)+ 0.5; \n"
        + "  aCoord = (uSTMatrix * vec4(coord,0.0,1.0)).xy;\n"
        + "}\n";

    private final String fragmentShaderCode =
        "#extension GL_OES_EGL_image_external : require\n"
        + "precision mediump float;\n"
        + "varying vec2 aCoord;\n"
        + "uniform samplerExternalOES vTexture;\n"
        + "void main() {\n"
        + "  gl_FragColor = texture2D(vTexture, aCoord);\n"
        + "}\n";
    private float[] stMatrix = {
        1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
    };

    private int stMatrixHandle;
    private int loadShader(int type, String shaderCode) {
        int shader = GLES20.glCreateShader(type);
        GLES20.glShaderSource(shader, shaderCode);
        GLES20.glCompileShader(shader);
        return shader;
    }
    public void setMatrix(float[] matrix) { stMatrix = matrix; }
    public Boolean init() {
        if (initStatus) {
            return initStatus;
        }
        ByteBuffer bb = ByteBuffer.allocateDirect(POSITION_VERTEX.length * 4);
        vertextBuffer = bb.order(ByteOrder.nativeOrder()).asFloatBuffer();
        vertextBuffer.put(POSITION_VERTEX);
        vertextBuffer.position(0);
        ByteBuffer m_bb = ByteBuffer.allocateDirect(TEXTURE_VERTEX.length * 4);
        texcoordBuffer = m_bb.order(ByteOrder.nativeOrder()).asFloatBuffer();
        texcoordBuffer.put(TEXTURE_VERTEX);
        texcoordBuffer.position(0);
        //        Log.i(TAG,"vertexShaderCode:" + vertexShaderCode);
        //        Log.i(TAG, "fragmentShaderCode:" + fragmentShaderCode);
        vertexShader = loadShader(GLES20.GL_VERTEX_SHADER, vertexShaderCode);
        fragmentShader =
            loadShader(GLES20.GL_FRAGMENT_SHADER, fragmentShaderCode);
        glShaderProgram = GLES20.glCreateProgram();

        GLES20.glAttachShader(glShaderProgram, vertexShader);
        GLES20.glAttachShader(glShaderProgram, fragmentShader);
        GLES20.glLinkProgram(glShaderProgram);
        GLES20.glDeleteShader(vertexShader);
        GLES20.glDeleteShader(fragmentShader);
        TextureUtils.LogOESError(TAG);
        stMatrixHandle =
            GLES20.glGetUniformLocation(glShaderProgram, "uSTMatrix");
        positiveHandler =
            GLES20.glGetAttribLocation(glShaderProgram, "vPosition");
        coordHandler = GLES20.glGetAttribLocation(glShaderProgram, "vCoord");
        frameTexture = GLES20.glGetUniformLocation(glShaderProgram, "vTexture");

        if (GLES20.glGetError() == GLES20.GL_NO_ERROR) {
            initStatus = true;
            return true;
        }
        initStatus = false;
        return false;
    }
    public int process(int input_texture, int output_texture, int width,
                       int height) {
        GLES20.glViewport(0, 0, width, height);
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, input_texture);
        GLES20.glUseProgram(glShaderProgram);
        if (frameBuffer == -1) {
            int[] frameBuffers = new int[1];
            GLES20.glGenFramebuffers(1, frameBuffers, 0);
            frameBuffer = frameBuffers[0];
        }
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, frameBuffer);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, output_texture);
        GLES20.glFramebufferTexture2D(GLES20.GL_FRAMEBUFFER,
                                      GLES20.GL_COLOR_ATTACHMENT0,
                                      GLES20.GL_TEXTURE_2D, output_texture, 0);
        int val = GLES20.glCheckFramebufferStatus(GLES20.GL_FRAMEBUFFER);
        if (val != GLES20.GL_FRAMEBUFFER_COMPLETE) {
            Log.e(TAG, "glFramebufferTexture2D failed,error code:" + val);
        }
        GLES20.glEnableVertexAttribArray(positiveHandler);
        GLES20.glEnableVertexAttribArray(coordHandler);
        GLES20.glUniform1i(frameTexture, 0);
        GLES20.glUniformMatrix4fv(stMatrixHandle, 1, false, stMatrix, 0);
        GLES20.glVertexAttribPointer(positiveHandler, 2, GLES20.GL_FLOAT, false,
                                     8, vertextBuffer);
        GLES20.glVertexAttribPointer(coordHandler, 2, GLES20.GL_FLOAT, false, 8,
                                     texcoordBuffer);
        GLES20.glEnableVertexAttribArray(positiveHandler);
        GLES20.glEnableVertexAttribArray(coordHandler);
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_FAN, 0, 4);
        GLES20.glDisableVertexAttribArray(positiveHandler);
        GLES20.glDisableVertexAttribArray(coordHandler);
        GLES20.glFramebufferTexture2D(GLES20.GL_FRAMEBUFFER,
                                      GLES20.GL_COLOR_ATTACHMENT0,
                                      GLES20.GL_TEXTURE_2D, 0, 0);
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0);
        //        TextureUtils.LogOESError(TAG);
        return 0;
    }
    @Override
    protected void finalize() {
        if (0 != glShaderProgram) {
            GLES20.glDeleteProgram(glShaderProgram);
        }
        if (0 != frameBuffer) {
            GLES20.glDeleteFramebuffers(1, new int[] {frameBuffer}, 0);
            frameBuffer = 0;
        }
    }
}
