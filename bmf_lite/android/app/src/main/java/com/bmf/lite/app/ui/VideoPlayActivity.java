package com.bmf.lite.app.ui;
import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.view.ViewGroup;
import android.widget.SeekBar;
import androidx.appcompat.app.AppCompatActivity;

import com.bmf.lite.app.tool.FileOperation;
import com.bmf.lite.app.R;
import com.bmf.lite.app.render.GLSurfaceViewWrapper;
import com.bmf.lite.app.playctrl.MediaPlayerWrapper;
import java.io.IOException;

public class VideoPlayActivity extends AppCompatActivity {
    private String TAG = "AlgRenderer VideoActivity";
    private GLSurfaceViewWrapper glSurfaceView = null;
    private String videoPath = "/sdcard/Camera/video0_720_1080.mp4";
    private MediaPlayerWrapper mediaPlayerWrapper = null;
    private boolean defaultVideoOpenSts = false;
    Thread defaultVideoPlayThread = null;
    Context context;
    private SeekBar splitScreenSeek;
    private float seekPosRatio = 0;
    int algType = 0;
    int playSts = 0;
    private void initView() {
        splitScreenSeek = (SeekBar)findViewById(R.id.splitSeekbar);
        glSurfaceView = findViewById(R.id.surfaceView);
        Intent intent = getIntent();
        algType = intent.getIntExtra("ALG_TYPE", 0);
        ViewGroup.MarginLayoutParams surfaceUiParams =
            (ViewGroup.MarginLayoutParams)glSurfaceView.getLayoutParams();
        ViewGroup.MarginLayoutParams seekBarParams =
            (ViewGroup.MarginLayoutParams)splitScreenSeek.getLayoutParams();
        seekBarParams.bottomMargin = -surfaceUiParams.height / 2;
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
                    Log.d(TAG, "seekbak start move ");
                }
                @Override
                public void onStopTrackingTouch(SeekBar seekBar) {
                    Log.d(TAG, "seekbak stop move ");
                    seekBar.setClickable(false);
                }
            });
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_player);
        context = this.getApplicationContext();
        initView();
        //           initPlay();
    }

    private void initPlay() { OpenDefaultVideo(); }
    private void OpenDefaultVideo() {
        if (playSts == 0) {
            if (defaultVideoPlayThread != null) {
                defaultVideoPlayThread.interrupt();
            }
            defaultVideoPlayThread = new Thread(new Runnable() {
                @Override
                public void run() {
                    try {
                        String defaultAssetVideoPath = "test.mp4";
                        String defaultVideoPath =
                            FileOperation.getAssetFilePath(
                                context, defaultAssetVideoPath);
                        Log.d(TAG, "videoPath:" + defaultVideoPath);
                        videoPath = defaultVideoPath;

                        if (null != mediaPlayerWrapper) {
                            mediaPlayerWrapper.pause();
                            mediaPlayerWrapper.stop();
                            mediaPlayerWrapper.destroy();
                        }
                        mediaPlayerWrapper =
                            new MediaPlayerWrapper(glSurfaceView);
                        while (glSurfaceView.getWndWidth() <= 0 ||
                               glSurfaceView.getWndHeight() <= 0) {
                            Thread.sleep(100);
                        }
                        mediaPlayerWrapper.startPlay(videoPath);
                        mediaPlayerWrapper.setLooping(true);
                        defaultVideoOpenSts = true;
                        mediaPlayerWrapper.switchAlgorithm(algType);
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                    Log.d(TAG, "videoPath:" + videoPath);
                }
            });
            defaultVideoPlayThread.start();
            playSts = 1;
        }
    }

    public void playLocalVideo(String videoPath) throws IOException {
        this.videoPath = videoPath;
        Log.d(TAG, "videoPath:" + videoPath);
        if (defaultVideoOpenSts == true && defaultVideoPlayThread != null) {
            defaultVideoPlayThread.interrupt();
        }
        if (null != mediaPlayerWrapper) {
            mediaPlayerWrapper.pause();
            mediaPlayerWrapper.stop();
            mediaPlayerWrapper.destroy();
        }
        mediaPlayerWrapper = new MediaPlayerWrapper(glSurfaceView);
        mediaPlayerWrapper.startPlay(this.videoPath);
        mediaPlayerWrapper.setLooping(true);
        Log.d(TAG, "videoPath:" + videoPath);
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (mediaPlayerWrapper == null) {
            initPlay();
        } else {
            mediaPlayerWrapper.resume();
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (mediaPlayerWrapper != null) {
            mediaPlayerWrapper.pause();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (mediaPlayerWrapper != null) {
            mediaPlayerWrapper.destroy();
        }
        if (glSurfaceView != null) {
            glSurfaceView.closeRender();
        }
    }
    @Override
    public void onActivityResult(int requestCode, int resultCode, Intent data) {
        try {
            super.onActivityResult(requestCode, resultCode, data);
            if (resultCode != Activity.RESULT_OK) {
                return;
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
