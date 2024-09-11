package com.bmf.lite.common;

import android.util.Log;

public class Logging {
    private static final String LOG_TAG = "bmf_lite";

    static public void d(String msg) { Log.d(LOG_TAG, msg); }

    public static void e(String msg) { Log.e(LOG_TAG, msg); }
}
