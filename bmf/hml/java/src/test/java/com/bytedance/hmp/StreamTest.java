package com.bytedance.hmp;

import junit.framework.TestCase;

public class StreamTest extends TestCase {
	public void testCudaStreamCreate()
    {
        if(Device.hasCuda()){
            Stream st0 = Stream.current(DeviceType.kCUDA);
            assertTrue(st0.query());
            assertTrue(st0.handle() == 0); //default stream
            assertTrue(st0.deviceType() == DeviceType.kCUDA);
            assertTrue(st0.deviceIndex() == 0);
            assertTrue(st0.own);

            Stream st1 = new Stream(DeviceType.kCUDA, 0);
            assertTrue(st1.handle() != 0);
            assertTrue(st1.deviceType() == DeviceType.kCUDA);
            assertTrue(st1.deviceIndex() == 0);
            
            Stream.setCurrent(st1);
            Stream st2 = Stream.current(DeviceType.kCUDA);
            assert(st2.handle() == st1.handle());

            Stream.setCurrent(st0);

            st0.free();
            st1.free();
            st2.free();
        }
    }

    public void testStreamGuard()
    {
        if(Device.hasCuda()){
            Stream st0 = Stream.current(DeviceType.kCUDA);
            Stream st1 = new Stream(DeviceType.kCUDA, 0);

            Stream.Guard guard = st1.new Guard();
            Stream st2 = Stream.current(DeviceType.kCUDA);
            assertTrue(st2.handle() == st1.handle());

            st0.free();
            st1.free();
            st2.free();
            guard.free();
        }
    }


	public void testCudaStreamAsyncExecution()
    {

    }
}
