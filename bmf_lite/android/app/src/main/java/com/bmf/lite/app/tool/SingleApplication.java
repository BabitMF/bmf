package com.bmf.lite.app.tool;

import android.app.Application;
import android.content.res.Resources;

public class SingleApplication {
    private static SingleApplication instance;
    private Application application;

    public static SingleApplication getInstance() {
        if (null == instance) {
            instance = new SingleApplication();
        }
        return instance;
    }
    public void init(Application application) {
        this.application = application;
    }
    public Application getContext() { return this.application; }
    public Resources getResources() { return this.application.getResources(); }
}
