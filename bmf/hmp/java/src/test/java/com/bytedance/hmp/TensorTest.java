package com.bytedance.hmp;

import junit.framework.TestCase;

public class TensorTest extends TestCase{
	public void testFactories()
    {
        {
            long[] shape = new long[] { 2, 3, 4 };
            Tensor t0 = Tensor.empty(shape, ScalarType.kFloat32, "cpu", false);
            assertTrue(t0.own);

            assertTrue(t0.toString().length() > 0);
            assertTrue(t0.defined());
            assertTrue(t0.dim() == 3);
            assertTrue(t0.size(0) == 2);
            assertTrue(t0.size(1) == 3);
            assertTrue(t0.size(2) == 4);
            assertTrue(t0.stride(0) == 12);
            assertTrue(t0.stride(1) == 4);
            assertTrue(t0.stride(2) == 1);
            assertTrue(t0.nitems() == 24);
            assertTrue(t0.itemsize() == 4);
            assertTrue(t0.nbytes() == 24 * 4);
            assertTrue(t0.dtype() == ScalarType.kFloat32);
            assertTrue(t0.isContiguous());
            assertTrue(t0.deviceType() == DeviceType.kCPU);
            assertTrue(t0.deviceIndex() == 0);

            t0.free();
        }

        {
            long[] shape = new long[] { 2, 3, 4 };
            Tensor t1 = Tensor.empty(shape, ScalarType.kFloat32, "cpu", false);

            t1.fill(1); //
            t1.fill(1.);
            t1.fill(true);

            t1.free();
        }

        {
            Tensor t2 = Tensor.arange(0, 100, 2, ScalarType.kInt32, "cpu", false);
            assertTrue(t2.dim() == 1);
            assertTrue(t2.size(0) == 50);
            t2.free();
        }

    }

	public void testCloneAlias()
    {
        long[] shape = new long[] { 2, 3, 4 };
        Tensor t0 = Tensor.empty(shape, ScalarType.kFloat32, "cpu", false);

        Tensor t1 = t0.clone();
        Tensor t2 = t0.alias();

        assertTrue(t1.dataPtr() != t0.dataPtr());
        assertTrue(t2.dataPtr() == t0.dataPtr());

        t0.free();
        t1.free();
        t2.free();
    }

	public void testViewReshape()
    {
        long[] shape = new long[] { 2, 3, 4 };
        Tensor t0 = Tensor.empty(shape, ScalarType.kFloat32, "cpu", false);

        Tensor t1 = t0.view(new long[]{4, 6});
        Tensor t2 = t0.reshape(new long[]{4, 6});

        assertTrue(t1.dataPtr() == t0.dataPtr());
        assertTrue(t2.dataPtr() == t0.dataPtr());

        t0.free();
        t1.free();
        t2.free();
    }

    public void testSliceSelect()
    {
        long[] shape = new long[] { 2, 3, 4 };
        Tensor t0 = Tensor.empty(shape, ScalarType.kFloat32, "cpu", false);

        Tensor t1 = t0.slice(1, 0, 3, 2);
        assertTrue(!t1.isContiguous());
        assertTrue(t1.size(0) == 2);
        assertTrue(t1.size(1) == 2);
        assertTrue(t1.size(2) == 4);

        Tensor t2 = t0.select(1, 0);
        assertTrue(t2.dim() == 2);
        assertTrue(t2.size(0) == 2);
        assertTrue(t2.size(1) == 4);

        t0.free();
        t1.free();
        t2.free();
    }


    public void testPermute()
    {
        long[] shape = new long[] { 2, 3, 4 };
        Tensor t0 = Tensor.empty(shape, ScalarType.kFloat32, "cpu", false);

        Tensor t1 = t0.permute(new long[]{2, 1, 0});
        assertTrue(t1.size(0) == 4);
        assertTrue(t1.size(1) == 3);
        assertTrue(t1.size(2) == 2);

        t0.free();
        t1.free();
    }

    public void testSqueeze()
    {
        long[] shape = new long[] { 2, 1, 4 };
        Tensor t0 = Tensor.empty(shape, ScalarType.kFloat32, "cpu", false);

        Tensor t1 = t0.squeeze(1);
        assertTrue(t1.dim() == 2);
        assertTrue(t1.size(0) == 2);
        assertTrue(t1.size(1) == 4);

        Tensor t2 = t1.unsqueeze(0);
        assertTrue(t2.dim() == 3);
        assertTrue(t2.size(0) == 1);
        assertTrue(t2.size(1) == 2);
        assertTrue(t2.size(2) == 4);

        t0.free();
        t1.free();
        t2.free();
    }

    public void testDataCopy()
    {
        {
            long[] shape = new long[] { 2, 3, 4 };
            Tensor t0 = Tensor.empty(shape, ScalarType.kFloat32, "cpu", false);

            Tensor t1 = t0.to(ScalarType.kUInt16);
            assertTrue(t1.dtype() == ScalarType.kUInt16);

            t1.copyFrom(t0); //expect no error

            t0.free();
            t1.free();
        }
    }

    public void testCudaTensor()
    {
        if(Device.hasCuda()){
            long[] shape = new long[] { 2, 3, 4 };
            Tensor t0 = Tensor.empty(shape, ScalarType.kFloat32, "cuda:0", false);
            
            assertTrue(t0.deviceType() == DeviceType.kCUDA);
            assertTrue(t0.deviceIndex() == 0);

            Tensor t1 = t0.to("cpu", true);
            assertTrue(t1.deviceType() == DeviceType.kCPU);

            t0.free();
            t1.free();
        }


    }

}
