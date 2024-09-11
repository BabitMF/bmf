package com.bmf.lite.app.playctrl;

import androidx.annotation.NonNull;
import android.content.Context;
import android.graphics.ImageFormat;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.CaptureResult;
import android.hardware.camera2.TotalCaptureResult;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.ImageReader;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Log;
import android.util.Size;
import android.view.Surface;

import com.bmf.lite.app.render.GLSurfaceViewWrapper;
import com.bmf.lite.app.tool.SingleApplication;

import java.util.Arrays;

public class CameraPlayerWrapper {
    private final String TAG = "bmf-demo-app CameraPlayerWrapper";
    private CameraManager cameraManager;
    private CameraDevice cameraDevice;
    private CameraCaptureSession captureSession;
    private CaptureRequest previewRequest;
    private CaptureRequest.Builder previewRequestBuilder;
    private Surface surface;
    private String cameraIdStr;
    private Handler backgroundHandler;
    private final String BACK_CAMERA = "0";
    private final String FRONT_CAMERA = "1";
    private GLSurfaceViewWrapper glSurfaceView;
    private int previewWidth = 0;
    private int previewHeight = 0;
    private int defaultWidth = 1080;
    private int defaultHeight = 1920;
    public CameraPlayerWrapper(GLSurfaceViewWrapper surfaceView) {
        glSurfaceView = surfaceView;
        glSurfaceView.setEGLContextClientVersion(3);
        initView(surfaceView);
        cameraIdStr = BACK_CAMERA;
    }

    private void initView(GLSurfaceViewWrapper glSurfaceView) {
        glSurfaceView.initRender();
        glSurfaceView.setOesStatus(true);
        glSurfaceView.setSplitScreenMode(1);
    }

    public void setPreviewSize(int width, int height) {
        previewWidth = width;
        previewHeight = height;
    }
    public void openCamera(int cameraId) {
        if (cameraId == 0) {
            cameraIdStr = BACK_CAMERA;
        } else {
            cameraIdStr = FRONT_CAMERA;
        }
        setUpCameraOutputs();
        try {
            Log.v(TAG, "openCamera cameraIdStr=" + cameraIdStr);
            cameraManager.openCamera(cameraIdStr, stateCallback,
                                     backgroundHandler);
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
    }

    @SuppressWarnings("SuspiciousNameCombination")
    private void setUpCameraOutputs() {
        cameraManager = (CameraManager)SingleApplication.getInstance()
                            .getContext()
                            .getSystemService(Context.CAMERA_SERVICE);
    }

    private final CameraDevice
        .StateCallback stateCallback = new CameraDevice.StateCallback() {
        @Override
        public void onOpened(@NonNull CameraDevice device) {
            try {
                SurfaceTexture surfaceTexture = null;
                while (true) {
                    surfaceTexture = glSurfaceView.getVideoRender()
                                         .getAlgRender()
                                         .getSurfaceTexture();
                    if (surfaceTexture != null) {
                        break;
                    }
                    Thread.sleep(1);
                }
                previewWidth = glSurfaceView.getWndWidth();
                previewHeight = glSurfaceView.getWndHeight();
                if (previewWidth <= 0 || previewHeight <= 0) {
                    Log.e(TAG, "current preview width:" + previewWidth +
                                   " height:" + previewHeight +
                                   ", camera open fail.");
                    return;
                }
                CameraCharacteristics cameraCharacteristics =
                    cameraManager.getCameraCharacteristics(cameraIdStr);
                StreamConfigurationMap streamConfigurationMap =
                    cameraCharacteristics.get(
                        CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
                Size[] sizes =
                    streamConfigurationMap.getOutputSizes(SurfaceTexture.class);
                Size size = CameraUtils.getInstance().getMostSuitableSize(
                    sizes, previewWidth, previewHeight);

                int orientation = cameraCharacteristics.get(
                    CameraCharacteristics.SENSOR_ORIENTATION);
                Log.d(TAG, "current sample size width:" + size.getWidth() +
                               " height:" + size.getHeight() +
                               ". orientation " + orientation);
                surfaceTexture.setDefaultBufferSize(size.getWidth(),
                                                    size.getHeight());

                if (orientation == 90 || orientation == 270) {
                    glSurfaceView.setRenderImgSize(size.getHeight(),
                                                   size.getWidth());
                } else {
                    glSurfaceView.setRenderImgSize(size.getWidth(),
                                                   size.getHeight());
                }

                surfaceTexture.setOnFrameAvailableListener(
                    new SurfaceTexture.OnFrameAvailableListener() {
                        @Override
                        public void onFrameAvailable(
                            final SurfaceTexture surfaceTexture) {
                            glSurfaceView.requestRender();
                        }
                    });
                surface = new Surface(surfaceTexture);
                cameraDevice = device;
                previewRequestBuilder =
                    device.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
                previewRequestBuilder.addTarget(surface);
                previewRequest = previewRequestBuilder.build();
                device.createCaptureSession(Arrays.asList(surface),
                                            sessionsStateCallback, null);
            } catch (CameraAccessException e) {
                e.printStackTrace();
                Log.e(TAG, "Open Camera Failed!");
            } catch (Exception e) {
                e.printStackTrace();
                Log.e(TAG, "Thread problem!");
            }
        }

        @Override
        public void onDisconnected(@NonNull CameraDevice device) {
            Log.v(TAG, "stateCallback onDisconnected()");
            device.close();
            device = null;
        }

        @Override
        public void onError(@NonNull CameraDevice device, int error) {
            Log.v(TAG, "stateCallback onError()");
            device.close();
            device = null;
        }
    };

    CameraCaptureSession.StateCallback sessionsStateCallback =
        new CameraCaptureSession.StateCallback() {
            @Override
            public void onConfigured(CameraCaptureSession session) {
                if (null == cameraDevice) {
                    return;
                }
                captureSession = session;
                try {
                    // Auto focus should be continuous for camera preview.
                    previewRequestBuilder.set(
                        CaptureRequest.CONTROL_AF_MODE,
                        CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE);

                    previewRequest = previewRequestBuilder.build();
                    Log.v(TAG, "sessionsStateCallback setRepeatingRequest()");
                    captureSession.setRepeatingRequest(
                        previewRequest, mCaptureCallback, backgroundHandler);
                } catch (CameraAccessException e) {
                    e.printStackTrace();
                }
            }

            @Override
            public void onConfigureFailed(CameraCaptureSession session) {
                Log.d(TAG, "onConfigureFailed!");
            }
        };

    private CameraCaptureSession.CaptureCallback mCaptureCallback =
        new CameraCaptureSession.CaptureCallback() {
            @Override
            public void onCaptureProgressed(
                @NonNull CameraCaptureSession session,
                @NonNull CaptureRequest request,
                @NonNull CaptureResult partialResult) {
                //            Log.v(TAG, "CameraCaptureSession
                //            onCaptureProgressed()");
            }

            @Override
            public void onCaptureCompleted(
                @NonNull CameraCaptureSession session,
                @NonNull CaptureRequest request,
                @NonNull TotalCaptureResult result) {
                //            Log.v(TAG, "CameraCaptureSession
                //            onCaptureCompleted()");
            }
        };

    private final ImageReader
        .OnImageAvailableListener onImageAvailableListener =
        new ImageReader.OnImageAvailableListener() {
            @Override
            public void onImageAvailable(ImageReader reader) {
                Log.v(TAG, "onImageAvailable()");
            }
        };

    public void closeCamera() {
        if (null != captureSession) {
            captureSession.close();
            captureSession = null;
        }
        if (null != cameraDevice) {
            cameraDevice.close();
            cameraDevice = null;
        }
    }

    public void switchAlgorithm(int algType) {
        if (glSurfaceView != null) {
            glSurfaceView.switchAlgorithm(algType);
        }
    }
}
