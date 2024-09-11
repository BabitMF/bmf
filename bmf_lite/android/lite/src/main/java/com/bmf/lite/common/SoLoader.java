package com.bmf.lite.common;

public class SoLoader {
    private static boolean sIsSoInitialized = false;
    private static final SoLoader INSTANCE = new SoLoader();

    static {
        try {
            System.loadLibrary("bmf_lite_jni");
            sIsSoInitialized = true;
            //            Logging.d("Bmf so libraries are initialized. version =
            //            " + BuildConfig.VERSION_NAME);
        } catch (Throwable t) {
            sIsSoInitialized = false;
        }
    }

    private SoLoader() {}
    public static SoLoader getInstance() { return INSTANCE; }
    public boolean isSoInitialized() { return sIsSoInitialized; }
}
