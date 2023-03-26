package com.bytedance.hmp;

import junit.framework.Test;
import junit.framework.TestCase;
import junit.framework.TestSuite;

/**
 * Unit test for simple App.
 */
public class DeviceTest 
    extends TestCase
{
    /**
     * Create the test case
     *
     * @param testName name of the test case
     */
    public DeviceTest( String testName )
    {
        super( testName );
    }

    /**
     * @return the suite of tests being tested
     */
    public static Test suite()
    {
        return new TestSuite( DeviceTest.class );
    }

    /**
     * Rigourous Test :-)
     */
    public void testDeviceCreate()
    {
        // cpu
        {
            Device d0 = new Device();
            assertTrue(d0.type() == DeviceType.kCPU);

            Device d1 = new Device("cpu");
            assertTrue(d1.type() == DeviceType.kCPU);
            assertTrue(d1.index() == 0);

            assertTrue(d0.equals(d1));

            assertTrue(Device.count(DeviceType.kCPU) == 1);

            assertTrue(d0.own);
            assertTrue(d1.own);

            d0.free();
            d1.free();
        }

        if(Device.count(DeviceType.kCUDA) > 0){
            Device d0 = new Device("cuda");
            assertTrue(d0.type() == DeviceType.kCUDA);
            assertTrue(d0.index() == 0);
            assertTrue(d0.toString().equals("cuda:0"));

            Device d1 = new Device("cuda:0");
            assertTrue(d1.type() == DeviceType.kCUDA);
            assertTrue(d1.index() == 0);
            assertTrue(d1.toString().equals("cuda:0"));

            assertTrue(d0.equals(d1));

            assertTrue(d0.own);
            assertTrue(d1.own);

            d0.free();
            d1.free();
        }
    }


    public void testDeviceGuard()
    {
        if(Device.count(DeviceType.kCUDA) > 0){
            Device d0 = new Device("cuda");
            Device.Guard guard = d0.new Guard();
            
            //TODO check

            guard.free();
            d0.free();

        }
    }
}
