package com.bytedance.hmp;

public enum ChannelFormat {
	kNCHW(0), kNHWC(1);

    private final int value;

    ChannelFormat(final int v){ value = v; }

    public int getValue() { return value; }
}
