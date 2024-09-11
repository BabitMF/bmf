package com.bmf.lite.app.playctrl;
import android.graphics.SurfaceTexture;
import android.media.MediaPlayer;
import android.net.Uri;
import android.util.Log;
import android.view.Surface;

import com.bmf.lite.app.render.GLSurfaceViewWrapper;
import com.bmf.lite.app.tool.SingleApplication;
import com.bmf.lite.app.tool.SingleApplication;
import java.io.IOException;

public class MediaPlayerWrapper {
    private boolean loopPlaySts = true;
    private GLSurfaceViewWrapper glSurfaceView = null;
    private String videoPath =
        ""; /// storage/emulated/0/DCIM/Camera/VID_20230307_104326.mp4";
    private String TAG = "bmf-demo-app MediaPlayerWrapper";
    private Surface surface = null;
    MediaPlayer mediaPlayer = null;
    public MediaPlayerWrapper(GLSurfaceViewWrapper surfaceView) {
        glSurfaceView = surfaceView;
        glSurfaceView.setOesStatus(true);
        glSurfaceView.initRender();
        glSurfaceView.setSplitScreenMode(1);
    }

    public void setOesStatus(boolean isOes) {
        glSurfaceView.setOesStatus(isOes);
    }

    public void startPlay(String videoPath) throws IOException {
        if (null == mediaPlayer) {
            mediaPlayer = new MediaPlayer();
            //            mediaPlayer =
            //            MediaPlayer.create(SingleApplication.getInstance().getContext(),
            //            Uri.parse(videoPath));
        }

        try {
            mediaPlayer.setDataSource(videoPath);
        } catch (IOException e) {
            e.printStackTrace();
        }
        mediaPlayer.setOnPreparedListener(new MediaPlayer.OnPreparedListener() {
            @Override
            public void onPrepared(MediaPlayer mp) {
                mp.start();
                glSurfaceView.setRenderImgSize(mp.getVideoWidth(),
                                               mp.getVideoHeight());
            }
        });
        SurfaceTexture st = null;
        try {
            while (true) {
                st = glSurfaceView.getVideoRender()
                         .getAlgRender()
                         .getSurfaceTexture();
                if (st != null) {
                    if (surface == null) {
                        surface = new Surface(st);
                    }
                    Thread.sleep(1);
                    break;
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
            Log.e(TAG, "Thread problem!");
        }
        mediaPlayer.setSurface(surface);
        mediaPlayer.prepareAsync();
    }

    public void switchAlgorithm(int algType) {
        if (glSurfaceView != null) {
            glSurfaceView.switchAlgorithm(algType);
        }
    }
    public void stop() {
        if (mediaPlayer != null) {
            mediaPlayer.stop();
        }
    }

    public void resume() {
        if (mediaPlayer != null) {
            mediaPlayer.pause();
            mediaPlayer.start();
        }
    }

    public void pause() {
        if (mediaPlayer != null) {
            mediaPlayer.pause();
        }
    }

    public void setLooping(boolean isLoop) {
        loopPlaySts = isLoop;
        if (mediaPlayer != null) {
            mediaPlayer.setLooping(isLoop);
        }
    }

    public void destroy() {
        if (mediaPlayer != null) {
            mediaPlayer.release();
        }
    }
}
