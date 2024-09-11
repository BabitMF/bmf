package com.bmf.lite.common;

public class ErrorCode {
    public static final int SUCCESS = 0x0000;
    public static final int MODULE_INIT_FAILED = 0x1000;
    public static final int DEVICE_NOT_SUPPORT = 0x1100;
    public static final int GPU_NOT_SUPPORT = 0x1101;
    public static final int OPENCL_NOT_SUPPORT = 0x1102;
    public static final int LOAD_SO_FAIL = 0x1103;

    public static final int PROCESS_FAILED = 0x2000;
    public static final int EXECUTE_OPS_FAILED = 0x2100;

    public static final int MODULE_NOT_INIT = 0x2201;
    public static final int NOT_FLUSH = 0x2202;
    public static final int INVALID_PARAMETER = 0x2203;
    public static final int INVALID_DATA = 0x2204;
    public static final int CREATE_JNI_RESOURCE_FAIL = 0x3000;
    public static final int JNI_RESOURCE_NOT_INIT = 0x3001;
    public static final int INSUFFICIENT_GPU_MEMORY = 0x3002;
    public static final int INSUFFICIENT_CPU_MEMORY = 0x3003;
    public static final int TIMEOUT = 0x4000;
    public static final int HANG = 0x40001;
}