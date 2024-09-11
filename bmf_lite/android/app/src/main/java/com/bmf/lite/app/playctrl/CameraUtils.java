package com.bmf.lite.app.playctrl;

import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.params.StreamConfigurationMap;

import android.util.Log;
import android.util.Size;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;

import java.util.List;

public class CameraUtils {
    private static CameraUtils sInstance;

    private final String TAG = "bmf-demo-app CameraUtils";

    private final String BACK_CAMERA = "0";
    private final String FRONT_CAMERA = "1";

    public static CameraUtils getInstance() {
        if (sInstance == null) {
            synchronized (CameraUtils.class) {
                if (sInstance == null) {
                    sInstance = new CameraUtils();
                }
            }
        }
        return sInstance;
    }

    public String getCameraId() { return BACK_CAMERA; }

    public Size getMostSuitableSize(Size[] sizeList, int width, int height) {
        float targetRatio = (height * 1.f) / (width * 1.f);
        Size result = null;

        for (int j = 1; j < sizeList.length; j++) {
            if (result == null ||
                isMoreSuitable(result, sizeList[j], targetRatio)) {
                result = sizeList[j];
            }
        }
        int findNum = 3;
        int findIndex = 0;
        while (result.getWidth() > 720 && findIndex < findNum) {
            float ratio = getRatio(result);
            List<Size> optionList = new ArrayList<Size>();

            for (int j = 0; j < sizeList.length; j++) {
                float current_ratio = getRatio(sizeList[j]);
                if (Math.abs(ratio - current_ratio) < 0.05f &&
                    result != sizeList[j] &&
                    getArea(result) > getArea(sizeList[j])) {
                    optionList.add(sizeList[j]);
                }
            }
            if (optionList.size() > 0) {
                float maxArea = 0;
                for (int j = 0; j < optionList.size(); j++) {
                    float currentArea = getArea(optionList.get(j));
                    if (currentArea > maxArea) {
                        maxArea = currentArea;
                    }
                }
                for (int j = 0; j < sizeList.length; j++) {
                    float currentArea = getArea(sizeList[j]);
                    if (Math.abs(maxArea - currentArea) < 0.05f) {
                        result = sizeList[j];
                        break;
                    }
                }
            } else {
                break;
            }
        }
        return result;
    }

    private boolean isMoreSuitable(Size current, Size target,
                                   float targetRatio) {
        if (current == null) {
            return true;
        }
        float currentRatioDif = Math.abs(getRatio(current) - targetRatio);
        float targetRatioDif = Math.abs(getRatio(target) - targetRatio);
        return targetRatioDif < currentRatioDif ||
            (currentRatioDif == targetRatioDif &&
             (getArea(target) > getArea(current)));
    }

    private int getArea(Size size) {
        return size.getWidth() * size.getHeight();
    }

    private float getRatio(Size size) {
        return (size.getWidth() * 1.f) / (size.getHeight() * 1.f);
    }
}
