package com.bmf.lite.app.ui;

import android.Manifest;
import android.app.Activity;
import android.app.Application;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;

import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.bmf.lite.app.R;
import com.bmf.lite.app.tool.SingleApplication;

public class MainActivity extends Activity {
    private Button buttonSuperResolution;
    private Button buttonDenoise;
    private Button buttonTexGenPic;
    private final String TAG = "MainActivity";

    public final int REQUEST_CAMERA_PERMISSION = 101; // 1 或者101
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        buttonSuperResolution =
            (Button)findViewById(R.id.btnSuperResolutionTest);
        buttonDenoise = (Button)findViewById(R.id.btnDenoiseTest);
        buttonTexGenPic = (Button)findViewById(R.id.btnTexGenPicTest);
        Application application = getApplication();
        SingleApplication single_app = SingleApplication.getInstance();
        single_app.init(application);
        if (ContextCompat.checkSelfPermission(this,
                                              Manifest.permission.CAMERA) !=
                PackageManager.PERMISSION_GRANTED ||
            ContextCompat.checkSelfPermission(
                this, Manifest.permission.WRITE_EXTERNAL_STORAGE) !=
                PackageManager.PERMISSION_GRANTED ||
            ContextCompat.checkSelfPermission(
                this, Manifest.permission.READ_EXTERNAL_STORAGE) !=
                PackageManager.PERMISSION_GRANTED) {
            requestCameraPermission();
        } else {
            Log.d(TAG, "RequesPermissions: have camera permission");
        }
        buttonSuperResolution.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // Create a new Intent to start the new Activity
                Intent intent =
                    new Intent(MainActivity.this, VideoPlayActivity.class);
                //                intent.putExtra("ALG_TYPE",
                //                "SuperResolution"); // string
                intent.putExtra(
                    "ALG_TYPE",
                    1); // int,0->normal,1->superresolution,2->denoise,
                // 3->tex-gen-img
                startActivity(intent);
            }
        });
        buttonDenoise.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent =
                    new Intent(MainActivity.this, CameraPlayActivity.class);
                intent.putExtra("ALG_TYPE", 2);
                startActivity(intent);
            }
        });
        buttonTexGenPic.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent =
                    new Intent(MainActivity.this, TexGenPicActivity.class);
                intent.putExtra("ALG_TYPE", 3);
                startActivity(intent);
            }
        });
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
}