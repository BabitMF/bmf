package com.bmf.lite.app.tool;

import android.content.Context;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

public class FileOperation {
    public static String getAssetFilePath(Context context, String fileName) {
        copyFile(context, fileName);
        return context.getFilesDir().getAbsolutePath() + File.separator +
            fileName;
    }

    private static void copyFile(Context context, String fileName) {
        InputStream inputStream = null;
        OutputStream outputStream = null;
        try {
            File dataFile = new File(context.getFilesDir(), fileName);
            inputStream = context.getAssets().open(fileName);
            if (dataFile.length() == inputStream.available()) {
                return;
            }
            byte[] dataBuffer = new byte[2048];
            outputStream = new FileOutputStream(dataFile);

            int dataLength = inputStream.read(dataBuffer);
            while (dataLength > 0) {
                outputStream.write(dataBuffer, 0, dataLength);
                dataLength = inputStream.read(dataBuffer);
            }
            outputStream.flush();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (inputStream != null) {
                try {
                    inputStream.close();
                    inputStream = null;
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
            if (outputStream != null) {
                try {
                    outputStream.close();
                    outputStream = null;
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }
}
