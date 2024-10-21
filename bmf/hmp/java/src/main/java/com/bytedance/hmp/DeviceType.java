package com.bytedance.hmp;

public enum DeviceType{
    kCPU(0), kCUDA(1);

    private final int value;

    DeviceType(final int v){ value = v; }

    public int getValue() { return value; }
}

