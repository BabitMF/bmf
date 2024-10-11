package com.bmf.lite.app.ui;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.text.TextWatcher;
import android.util.Log;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import com.bmf.lite.app.R;
import com.bmf.lite.app.playctrl.ImagePlayerWrapper;
import com.bmf.lite.app.render.GLSurfaceViewWrapper;
import android.app.Activity;
import android.database.Cursor;
import android.net.Uri;
import android.provider.MediaStore;
import android.widget.Toast;
import android.text.Editable;
import com.bmf.lite.app.tool.FileOperation;

public class TexGenPicActivity extends AppCompatActivity {
    private final String TAG = "AlgRenderer TexGenPicActivity";
    private GLSurfaceViewWrapper glSurfaceView;
    private EditText inputEditView;
    private int algType = 0;
    int charSizeLimit = 30;
    private Button buttonChoiceImg;
    private Button buttonTexGenPic;
    private int algProgressSts = 0;
    private int lastAlgProgressSts = 0;
    private int imgLoadSts = 0;
    String imgPath;
    private ImagePlayerWrapper imagePlayerWrapper = null;
    private int modelLoadStsIndex = 0;
    String assetFilePath = null;
    String modelPath = "/data/local/tmp/ControlNetData";
    AlgListenThread listenAlgStsThread = null;

    public class AlgListenThread extends Thread {
        private volatile boolean running = true;
        public void stopCustomThread() { this.interrupt(); }

        @Override
        public void run() {
            try {
                while (running) {
                    if (null != imagePlayerWrapper) {
                        algProgressSts = imagePlayerWrapper.getAlgProgressSts();
                        if (algProgressSts == -200) {
                            buttonTexGenPic.setText("设备不支持");
                            buttonTexGenPic.setEnabled(false);
                            Toast
                                .makeText(TexGenPicActivity.this,
                                          "The current device does not "
                                              + "support Vincennes",
                                          Toast.LENGTH_SHORT)
                                .show();
                            break;
                        }
                    }
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            if (algProgressSts == -1000 && imgLoadSts == 1) {
                                if ((modelLoadStsIndex / 250) % 2 == 0) {
                                    buttonTexGenPic.setText("模型加载...");
                                } else {
                                    buttonTexGenPic.setText("模型加载....");
                                }
                                modelLoadStsIndex++;
                                if (modelLoadStsIndex >= 10000) {
                                    modelLoadStsIndex = 0;
                                }
                                inputEditView.setEnabled(false);
                            }
                            if (algProgressSts != lastAlgProgressSts &&
                                imgLoadSts == 1) {
                                if (algProgressSts < 99 && algProgressSts > 0) {
                                    String progressStr =
                                        "生成" + algProgressSts + "%..";
                                    Log.d(TAG,
                                          "current progressStr:" + progressStr);
                                    buttonTexGenPic.setText(progressStr);
                                    buttonTexGenPic.setEnabled(false);
                                    inputEditView.setEnabled(false);
                                } else if (algProgressSts == 100) {
                                    buttonTexGenPic.setText("生成完毕");
                                    inputEditView.setEnabled(true);
                                } else {
                                    if (algProgressSts == 0) {
                                        buttonTexGenPic.setText("生成图片");
                                        buttonTexGenPic.setEnabled(true);
                                        inputEditView.setEnabled(true);
                                    }
                                }
                            }
                        }
                    });
                    Thread.sleep(10);
                }
            } catch (InterruptedException e) {
                running = false;
                return;
            } catch (Exception e) {
                e.printStackTrace();
            }
        };
    }
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_player_tex_gen_pic);
        RequestFilePermissions();
        initView();
        if (imagePlayerWrapper == null) {
            imagePlayerWrapper = new ImagePlayerWrapper(glSurfaceView);
        }
        String[] algDependParams = {
            assetFilePath + ":/vendor/dsp/cdsp:/vendor/lib64:/",
            assetFilePath +
                (";/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/" +
                 "adsp;/vendor/dsp/dsp;/vendor/dsp/images;/dsp"),
            assetFilePath + "/libQnnHtp.so",
            assetFilePath + "/libQnnSystem.so",
            modelPath,
            modelPath + "/unet.serialized.bin",
            modelPath + "/text_encoder.serialized.bin",
            modelPath + "/vae_decoder.serialized.bin",
            modelPath + "/controlnet.serialized.bin",
        };
        glSurfaceView.setAlgDependParams(algDependParams);
    }
    public void startListentAlgSts() {
        if (listenAlgStsThread != null) {
            listenAlgStsThread.stopCustomThread();
        }
        listenAlgStsThread = new AlgListenThread();
        listenAlgStsThread.start();
    }
    public void initView() {
        inputEditView = (EditText)findViewById(R.id.inputEditText);
        glSurfaceView = findViewById(R.id.surfaceView);
        Intent intent = getIntent();
        // Get and use the passed data
        algType = intent.getIntExtra("ALG_TYPE", 0);
        Log.d(TAG, "current algType:" + algType);
        assetFilePath =
            this.getApplicationContext().getFilesDir().getAbsolutePath();
        Log.d(TAG, assetFilePath);

        Log.d(TAG, this.getApplicationContext().getApplicationInfo().dataDir);
        LinearLayout.LayoutParams layoutParams = new LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT, 0, 0.92f);
        glSurfaceView.setLayoutParams(layoutParams);

        LinearLayout.LayoutParams layoutParamsTex =
            new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, 0, 0.08f);
        inputEditView.setLayoutParams(layoutParamsTex);
        buttonChoiceImg = findViewById(R.id.importRefPicBtn);
        buttonTexGenPic = findViewById(R.id.generatePicBtn);
        buttonChoiceImg.setOnClickListener(v -> {
            if (algProgressSts == 0 || algProgressSts >= 100) {
                if (listenAlgStsThread != null) {
                    listenAlgStsThread.stopCustomThread();
                    listenAlgStsThread = null;
                }
                inputEditView.setEnabled(true);
                buttonTexGenPic.setEnabled(true);
                buttonTexGenPic.setText("生成图片");
                onChoiceImg();
            } else {
                Toast
                    .makeText(TexGenPicActivity.this,
                              "Model loading or generating..",
                              Toast.LENGTH_SHORT)
                    .show();
                return;
            }
        });

        buttonTexGenPic.setOnClickListener(v -> { onTexGenPicBty(); });

        inputEditView.addTextChangedListener(new TextWatcher() {
            int beforeLength = 0;
            int cursor = 0;
            @Override
            public void beforeTextChanged(CharSequence s, int start, int count,
                                          int after) {
                beforeLength = s.length();
            }
            @Override
            public void onTextChanged(CharSequence s, int start, int before,
                                      int count) {
                if (algProgressSts == 0 || algProgressSts >= 100) {
                    if (listenAlgStsThread != null) {
                        listenAlgStsThread.stopCustomThread();
                    }
                    buttonTexGenPic.setEnabled(true);
                    buttonTexGenPic.setText("生成图片");
                    modelLoadStsIndex = 0;
                } else {
                    Toast
                        .makeText(TexGenPicActivity.this,
                                  "Model loading or generating..",
                                  Toast.LENGTH_SHORT)
                        .show();
                    return;
                }
            }
            @Override
            public void afterTextChanged(Editable s) {
                Log.d("This is already entered", "" + s.length());
                int afterLength = s.length();
                if (afterLength > charSizeLimit) {
                    int dValue = afterLength - charSizeLimit;
                    int dNum = afterLength - beforeLength;
                    int st = cursor + (dNum - dValue);
                    int en = cursor + dNum;
                    Editable s_new = s.delete(st, en);
                    inputEditView.setText(s_new.toString());
                    inputEditView.setSelection(st);
                    Toast
                        .makeText(TexGenPicActivity.this,
                                  "The maximum word limit has been exceeded",
                                  Toast.LENGTH_SHORT)
                        .show();
                }
            }
        });
    }

    @Override
    public void onResume() {
        super.onResume();
    }

    @Override
    public void onPause() {
        super.onPause();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (glSurfaceView != null) {
            glSurfaceView.closeRender();
        }
    }

    protected void RequestFilePermissions() {
        if (ContextCompat.checkSelfPermission(
                this, Manifest.permission.WRITE_EXTERNAL_STORAGE) !=
            PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(
                TexGenPicActivity.this,
                new String[] {Manifest.permission.WRITE_EXTERNAL_STORAGE}, 100);
        } else {
            Log.d(TAG, "RequestFilePermissions: have read permission");
        }
        if (ContextCompat.checkSelfPermission(
                this, Manifest.permission.READ_EXTERNAL_STORAGE) !=
            PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(
                TexGenPicActivity.this,
                new String[] {Manifest.permission.READ_EXTERNAL_STORAGE}, 100);
        } else {
            Log.d(TAG, "RequestFilePermissions: have write permission.");
        }
    }

    public void onChoiceImg() {
        try {
            Intent intent =
                new Intent(Intent.ACTION_PICK,
                           MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
            startActivityForResult(intent, 66);
            modelLoadStsIndex = 0;
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void onTexGenPicBty() {
        String text = inputEditView.getText().toString();
        if (text == null) {
            Toast.makeText(TexGenPicActivity.this, "The input string is empty",
                          Toast.LENGTH_SHORT)
                .show();
        } else {
            if (imgLoadSts == 1) {
                glSurfaceView.setAlgInputParam(text);
                buttonTexGenPic.setText("模型加载...");
                modelLoadStsIndex++;
                buttonTexGenPic.setEnabled(false);
                inputEditView.setEnabled(false);
                startListentAlgSts();
            }
        }
    }

    public void initPlay(String resourcePath) {
        try {
            imgLoadSts = 0;
            int ret = imagePlayerWrapper.setCurImgResource(resourcePath);
            if (ret == 0) {
                imgLoadSts = 1;
                modelLoadStsIndex = 0;
            }
        } catch (Exception e) {
            e.printStackTrace();
            Log.e(TAG, "Thread problem!");
        }
        imagePlayerWrapper.switchAlgorithm(algType);
    }
    @Override
    public void onActivityResult(int requestCode, int resultCode, Intent data) {
        try {
            super.onActivityResult(requestCode, resultCode, data);
            if (requestCode == 66 && resultCode == RESULT_OK && null != data) {
                Uri selectedImage = data.getData();
                String[] filePathColumn = {MediaStore.Images.Media.DATA};
                Cursor cursor = getContentResolver().query(
                    selectedImage, filePathColumn, null, null, null);
                cursor.moveToFirst();
                int columnIndex = cursor.getColumnIndex(filePathColumn[0]);
                imgPath = cursor.getString(columnIndex);
                cursor.close();
                Toast
                    .makeText(TexGenPicActivity.this,
                              "select_img_path:" + imgPath, Toast.LENGTH_SHORT)
                    .show();
                initPlay(imgPath);
            }

            if (resultCode != Activity.RESULT_OK) {
                return;
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}