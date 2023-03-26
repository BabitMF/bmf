package com.bytedance.hmp;

public enum ScalarType {
	kUInt8(0), 
    kInt(1),
    kUInt16(2),
    kInt16(3),
    kInt32(4),
    kInt64(5),
    kFloat32(6),
    kFloat64(7),
    kHalf(8);

    int value;
    ScalarType(int v) { value = v; }

    public int getValue() { return value; }
}
