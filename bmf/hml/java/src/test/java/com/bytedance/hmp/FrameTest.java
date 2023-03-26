package com.bytedance.hmp;


import com.bytedance.hmp.ChannelFormat;
import com.bytedance.hmp.PixelFormat;

import junit.framework.TestCase;

public class FrameTest extends TestCase{
	public void testConstructors()
    {
        PixelInfo h420 = new PixelInfo(PixelFormat.PF_YUV420P,
                                       ColorSpace.CS_BT709,
                                       ColorRange.CR_MPEG);

        Frame f0 = new Frame(1920, 1080, h420, "cpu");
        assertTrue(f0.own);

        assertTrue(f0.defined());
        assertTrue(f0.pixInfo().format() == PixelFormat.PF_YUV420P);
        assertTrue(f0.format() == PixelFormat.PF_YUV420P);
        assertTrue(f0.width() == 1920);
        assertTrue(f0.height() == 1080);
        assertTrue(f0.dtype() == ScalarType.kUInt8);
        assertTrue(f0.deviceType() == DeviceType.kCPU);
        assertTrue(f0.deviceIndex() == 0);
        assertTrue(f0.nplanes() == 3);
        assertTrue(f0.toString().length() > 0);
        Tensor plane = f0.plane(0);
        assertFalse(plane.own);

        Tensor[] planes = new Tensor[]{f0.plane(0), f0.plane(1), f0.plane(2)};
        Frame f1 = new Frame(planes, h420);
        assert(f1.defined());
        assert(f1.own);

        Frame f2 = new Frame(planes, 1920, 1080, h420);
        assert(f2.defined());
        assert(f2.own);

        plane.free();
        h420.free();
        f0.free();
        f1.free();
        f2.free();
        planes[0].free();
        planes[1].free();
        planes[2].free();
    }

    public void testDataCopy()
    {
        PixelInfo h420 = new PixelInfo(PixelFormat.PF_YUV420P,
                                       ColorSpace.CS_BT709,
                                       ColorRange.CR_MPEG);

        Frame f0 = new Frame(1920, 1080, h420, "cpu");

        Frame f1 = f0.clone();
        assertTrue(f1.width() == 1920);
        assertTrue(f1.height() == 1080);
        f1.copyFrom(f0);
        assertTrue(f1.own);

        h420.free();
        f0.free();
        f1.free();

    }

    public void testCrop()
    {
        PixelInfo h420 = new PixelInfo(PixelFormat.PF_YUV420P,
                                       ColorSpace.CS_BT709,
                                       ColorRange.CR_MPEG);

        Frame f0 = new Frame(1920, 1080, h420, "cpu");

        Frame f1 = f0.crop(0, 0, 1280, 720);
        assertTrue(f1.width() == 1280);
        assertTrue(f1.height() == 720);
        assertTrue(f1.own);

        h420.free();
        f0.free();
        f1.free();
    }

    public void testCudaFrame()
    {
        if(Device.hasCuda()){
            PixelInfo h420 = new PixelInfo(PixelFormat.PF_YUV420P,
                                           ColorSpace.CS_BT709,
                                           ColorRange.CR_MPEG);

            Frame f0 = new Frame(1920, 1080, h420, "cuda:0");
            assertTrue(f0.deviceType() == DeviceType.kCUDA);
            assertTrue(f0.deviceIndex() == 0);
            Frame f1 = f0.to("cpu", false);
            assertTrue(f1.deviceType() == DeviceType.kCPU);
            assert(f1.own);

            h420.free();
            f0.free();
            f1.free();
        }
    }


    public void testImageConvert()
    {
        PixelInfo h420 = new PixelInfo(PixelFormat.PF_YUV420P,
                                        ColorSpace.CS_BT709,
                                        ColorRange.CR_MPEG);

        Frame f0 = new Frame(1920, 1080, h420, "cpu");
        Image im0 = f0.toImage(ChannelFormat.kNCHW);
        assertTrue(im0.width() == 1920);
        assertTrue(im0.height() == 1080);

        Frame f1 = Frame.fromImage(im0, h420);
        assertTrue(f1.width() == 1920);
        assertTrue(f1.height() == 1080);

        h420.free();
        im0.free();
        f0.free();
        f1.free();
    }
}
