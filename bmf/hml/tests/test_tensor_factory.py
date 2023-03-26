
import pytest
from hml_fixtures import device_type, has_cuda, dtype, to_np_dtype, with_half
from hml_fixtures import mp
import numpy as np

class TestTensorFactory(object):
    @pytest.fixture
    def with_half(self):
        return True

    def test_attrs(self, device_type, dtype):
        shape = (2, 3, 4, 5, 6)

        # default
        a = mp.empty(shape)
        assert(a.dtype == mp.kFloat32)
        assert(a.device_type == mp.kCPU)
        assert(a.shape == shape)
        assert(a.dim == len(shape))

        #
        a = mp.empty(shape, device=device_type, dtype=dtype)
        assert(a.device_type == device_type)
        assert(a.shape == shape)
        assert(a.dtype == dtype)
        assert(a.dim == len(shape))

        if dtype != mp.kHalf:
            b = np.empty(shape, dtype=to_np_dtype(dtype))
            assert(a.nbytes == b.nbytes)
            assert(a.itemsize == b.itemsize)
            assert(a.shape == b.shape)
            expect_strides = tuple([s//b.itemsize for s in b.strides])
            assert(a.strides == expect_strides)

            #
            a = a.select(1, 2).slice(1, 0, 2).slice(-1, 0, 6, 2)
            b = b[:, 2, :2, :, ::2]
            assert(a.nbytes == b.nbytes)
            assert(a.itemsize == b.itemsize)
            assert(a.shape == b.shape)
            expect_strides = tuple([s//b.itemsize for s in b.strides])
            assert(a.strides == expect_strides)


    def test_alias(self):
        a = mp.arange(0, 10)
        b = a.alias()
        b.slice(0, 5, 10).fill_(1)

        a_np = a.cpu().numpy().tolist()
        assert(a_np == [0, 1, 2, 3, 4, 1, 1, 1, 1, 1])


    def test_view(self):
        a = mp.arange(0, 12)
        a = a.reshape((3, 4))

        b = a.view((4, 3))
        b.select(0, 0).fill_(16)
        
        a_np = a.cpu().numpy().flatten().tolist()
        assert(a_np == [16, 16, 16, 3, 4, 5, 6, 7, 8, 9, 10, 11])

        a = a.slice(0, 0, 3, 2)
        assert(a.is_contiguous == False)
        b = a.view((2, 4)) # same shape, view ok
        with pytest.raises(RuntimeError):
            b = a.view((4, 2)) #stride can't be changed


    def test_clone(self):
        a = mp.arange(0, 5)
        b = a.clone().fill_(0)

        a_np = a.cpu().numpy().tolist()
        b_np = b.cpu().numpy().tolist()
        assert(a_np == [0, 1, 2, 3, 4])
        assert(b_np == [0, 0, 0, 0, 0])


    def test_as_strided(self):
        a = mp.arange(0, 5)
        b = a.as_strided_([2, 5], [0, 1])
        assert(a.shape == (2, 5))
        assert(b.shape == (2, 5))
        a_np = a.cpu().numpy().flatten().tolist()
        assert(a_np == [0, 1, 2, 3, 4, 0, 1, 2, 3, 4])

        a = mp.arange(0, 5)
        b = a.as_strided([2, 5], [0, 1])
        assert(a.shape == (5,))
        a_np = a.cpu().numpy().flatten().tolist()
        b_np = b.cpu().numpy().flatten().tolist()
        assert(a_np == [0, 1, 2, 3, 4])
        assert(b_np == [0, 1, 2, 3, 4, 0, 1, 2, 3, 4])


    def test_empty_like(self, device_type, dtype):
        a = mp.empty((1,), device=device_type, dtype=dtype)

        b = mp.empty_like(a)
        assert(b.shape == a.shape)
        assert(b.device == a.device)
        assert(b.dtype == a.dtype)

        c = mp.empty_like(a, device=mp.kCPU)
        assert(c.shape == a.shape)
        assert(c.device_type == mp.kCPU)
        assert(c.dtype == a.dtype)

        d = mp.empty_like(a, dtype=mp.kUInt8)
        assert(d.shape == a.shape)
        assert(d.device == a.device)
        assert(d.dtype == mp.kUInt8)


    def test_numpy(self, dtype):
        a = np.arange(1000, dtype=to_np_dtype(dtype))
        b = a.reshape((2, 4, 5, 25))
        c = b[1, ::2, ::2, ::5]
        ds = [a, b, c]

        a0 = mp.from_numpy(a).numpy()
        b0 = mp.from_numpy(b).numpy()
        c0 = mp.from_numpy(c).numpy()
        ds0 = mp.to_numpy(mp.from_numpy(ds))

        assert((a == a0).all())
        assert((b == b0).all())
        assert((c == c0).all())
        for i in range(len(ds)):
            assert((ds[i] == ds0[i]).all())

        a1 = mp.from_numpy(a).clone().numpy()
        b1 = mp.from_numpy(b).clone().numpy()
        c1 = mp.from_numpy(c).clone().numpy()

        assert((a == a1).all())
        assert((b == b1).all())
        assert((c == c1).all())


class TestTensorFactoryWithoutHalf(object):
    def test_arange(self, device_type, dtype):
        a = mp.arange(2, 12, device=device_type, dtype=dtype)
        b = mp.arange(2, 12, 2, device=device_type, dtype=dtype)
        c = mp.arange(2, 12, 3, device=device_type, dtype=dtype)

        a = a.cpu().numpy()
        b = b.cpu().numpy()
        c = c.cpu().numpy()

        a_expect = np.arange(2, 12, dtype=to_np_dtype(dtype))
        b_expect = np.arange(2, 12, 2, dtype=to_np_dtype(dtype))
        c_expect = np.arange(2, 12, 3, dtype=to_np_dtype(dtype))

        assert((a == a_expect).all())
        assert((b == b_expect).all())
        assert((c == c_expect).all())

        #irregular
        with pytest.raises(RuntimeError):
            mp.arange(2, 2, device=device_type, dtype=dtype)

        with pytest.raises(RuntimeError):
            mp.arange(2, 0, device=device_type, dtype=dtype)

        with pytest.raises(RuntimeError):
            mp.arange(2, 10, -1, device=device_type, dtype=dtype)

        with pytest.raises(RuntimeError):
            mp.arange(2, 10, 0, device=device_type, dtype=dtype)


    def test_arange_out_of_range(self, device_type):
        a = mp.arange(2, 1024, dtype=mp.kUInt8, device=device_type)
        a = a.cpu().numpy()
        a_expect = np.arange(2, 1024, dtype=np.uint8)
        assert((a == a_expect).all())

