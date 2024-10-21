package com.bytedance.hmp;

import com.bytedance.hmp.ScalarType;

import junit.framework.TestCase;

public class ImageTest extends TestCase{
    public void testConstructors()
    {
        Image im0 = new Image(1920, 1080, 3, ChannelFormat.kNCHW, ScalarType.kUInt8, "cpu", false);

        assertTrue(im0.defined());
        assertTrue(im0.wdim() == 2);
        assertTrue(im0.hdim() == 1);
        assertTrue(im0.cdim() == 0);
        assertTrue(im0.width() == 1920);
        assertTrue(im0.height() == 1080);
        assertTrue(im0.nchannels() == 3);
        assertTrue(im0.dtype() == ScalarType.kUInt8);
        assertTrue(im0.deviceType() == DeviceType.kCPU);
        assertTrue(im0.deviceIndex() == 0);
        Tensor d0 = im0.data();
        assertFalse(d0.own);
        assertTrue(d0.defined());
        assertTrue(im0.toString().length() > 0);

        im0.free();
        d0.free();
    }

    public void testCrop()
    {
        Image im0 = new Image(1920, 1080, 3, ChannelFormat.kNCHW, ScalarType.kUInt8, "cpu", false);
        Image im1 = im0.crop(0, 0, 1280, 720);

        assertTrue(im1.width() == 1280);
        assertTrue(im1.height() == 720);

        im0.free();
        im1.free();
    }

    public void testDataCopy()
    {
        Image im0 = new Image(1920, 1080, 3, ChannelFormat.kNCHW, ScalarType.kUInt8, "cpu", false);
        Image im1 = im0.clone();
        im1.copyFrom(im0);

        assertTrue(im1.defined());

        im0.free();
        im1.free();
    }

    public void testCudaImage()
    {
        if(Device.hasCuda()){
            Image im0 = new Image(1920, 1080, 3, ChannelFormat.kNCHW, 
                                  ScalarType.kUInt8, "cuda:0", false);
            assertTrue(im0.deviceType() == DeviceType.kCUDA);

            Image im1 = im0.to("cpu", true);
            assertTrue(im1.deviceType() == DeviceType.kCPU);

            im0.free();
            im1.free();
        }
    }


}
