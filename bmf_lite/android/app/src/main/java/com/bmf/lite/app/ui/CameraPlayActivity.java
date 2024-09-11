package com.bmf.lite.app.ui;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;
import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Point;
import android.os.Bundle;
import android.util.Log;
import android.view.Window;
import android.view.WindowManager;
import android.widget.SeekBar;
import com.bmf.lite.app.R;
import com.bmf.lite.app.playctrl.CameraPlayerWrapper;
import com.bmf.lite.app.render.GLSurfaceViewWrapper;

public class CameraPlayActivity extends AppCompatActivity {
    private final String TAG = "AlgRenderer CameraPlayActivity";
    private GLSurfaceViewWrapper glSurfaceView;
    public final int REQUEST_CAMERA_PERMISSION = 101; // 1 或者101
    private CameraPlayerWrapper cameraPlayer = null;
    private SeekBar splitScreenSeek;
    private float seekPosRatio = 0;
    int algType = 0;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_player);
        if (ContextCompat.checkSelfPermission(this,
                                              Manifest.permission.CAMERA) !=
            PackageManager.PERMISSION_GRANTED) {
            requestCameraPermission();
        } else {
            initView();
            //           initPlay();
        }
    }
    private void setWindowFlag() {
        Window window = getWindow();
        window.addFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN);
        window.addFlags(WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS);
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
    }

    public void initView() {
        splitScreenSeek = (SeekBar)findViewById(R.id.splitSeekbar);
        glSurfaceView = (GLSurfaceViewWrapper)findViewById(R.id.surfaceView);
        Intent intent = getIntent();
        // Get and use the passed data
        algType = intent.getIntExtra("ALG_TYPE", 0);
        Log.d(TAG, "current algType:" + algType);
        splitScreenSeek.setMax(100);
        splitScreenSeek.setMin(0);
        splitScreenSeek.setClickable(false);
        splitScreenSeek.setOnSeekBarChangeListener(
            new SeekBar.OnSeekBarChangeListener() {
                @Override
                public void onProgressChanged(SeekBar seekBar, int progress,
                                              boolean fromUser) {
                    seekPosRatio = (float)progress / 100.0f;
                    glSurfaceView.setSplitScreenPos(seekPosRatio);
                }
                @Override
                public void onStartTrackingTouch(SeekBar seekBar) {
                    Log.d(TAG, "seekbar start move ");
                }
                @Override
                public void onStopTrackingTouch(SeekBar seekBar) {
                    Log.d(TAG, "seekbar stop move ");
                    seekBar.setClickable(false);
                }
            });
    }

    public void initPlay() {
        if (cameraPlayer == null) {
            cameraPlayer = new CameraPlayerWrapper(glSurfaceView);
            if (glSurfaceView.getWndWidth() <= 0 ||
                glSurfaceView.getWndHeight() <= 0) {
                Log.d(TAG, "camera window is not ready");
            }

            cameraPlayer.switchAlgorithm(algType);
            cameraPlayer.openCamera(
                0); // 0->back camera,1->front
                    // cameracameraPlayer.switchAlgorithm(algType);
        }
    }

    @Override
    public void onResume() {
        super.onResume();
        initPlay();
    }

    @Override
    public void onPause() {
        super.onPause();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (cameraPlayer != null) {
            cameraPlayer.closeCamera();
        }
        if (glSurfaceView != null) {
            glSurfaceView.closeRender();
        }
    }

    public void requestCameraPermission() {
        if (shouldShowRequestPermissionRationale(Manifest.permission.CAMERA)) {
            Log.d(TAG, "requestCameraPermission: have camera permission");
        } else {
            requestPermissions(
                new String[] {Manifest.permission.CAMERA,
                              Manifest.permission.WRITE_EXTERNAL_STORAGE,
                              Manifest.permission.READ_EXTERNAL_STORAGE,
                              Manifest.permission.RECORD_AUDIO},
                REQUEST_CAMERA_PERMISSION);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode,
                                           String[] permissions,
                                           int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions,
                                         grantResults);
        if (requestCode == REQUEST_CAMERA_PERMISSION) {
            for (int i = 0; i < grantResults.length; i++) {
                if (grantResults[i] != PackageManager.PERMISSION_GRANTED) {
                    requestCameraPermission();
                    return;
                }
            }
            initView();
            initPlay();
        }
    }
}
