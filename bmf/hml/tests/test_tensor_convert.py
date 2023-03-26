import pytest
from hml_fixtures import device_type, has_cuda, dtype, f_dtype, to_np_dtype, with_half
from hml_fixtures import mp, scalar_types, device_types
import numpy as np

@pytest.fixture(params=[pytest.param(d, id=str(d)) for d in device_types if mp.device_count(d) > 0])
def device_type2(request):
    return request.param


@pytest.fixture(params=[pytest.param(t, id=repr(t)) for t in scalar_types])
def dtype2(request, with_half):
    if not with_half and request.param == mp.kHalf:
        pytest.skip("kHalf not support")
    else:
        return request.param


def test_tensor_dtype_convert(device_type, dtype, dtype2):
    a = np.random.uniform(size=1<<16)*1000
    a = a.astype(to_np_dtype(dtype))
    b = a.astype(to_np_dtype(dtype2))

    c = mp.from_numpy(a).to(device_type)
    d = c.to(dtype2).cpu().numpy()
    assert((b == d).all())

    b = a.reshape((4, 4, -1))[:, ::2, :].astype(to_np_dtype(dtype2))
    c = c.reshape((4, 4, -1)).slice(1, 0, 4, 2)
    assert(c.is_contiguous == False)
    d = c.to(dtype2).cpu().numpy()
    assert((b == d).all())


def test_tensor_device_convert(device_type, device_type2):
    a = np.random.uniform(size=1<<16).astype(np.float32)*1000

    c = mp.from_numpy(a).to(device_type)
    d = c.to(device_type2).cpu().numpy()
    assert((a == d).all())

    b = a.reshape((4, 4, -1))[:, ::2, ::3]
    c = c.reshape((4, 4, -1)).slice(1, 0, 4, 2).slice(2, 0, 4096, 3)
    assert(c.shape == b.shape)
    assert(c.is_contiguous == False)
    d = c.to(device_type2).cpu().numpy()
    assert((b == d).all())



class TestWithHalf(object):
    @pytest.fixture
    def with_half(self):
        return True


    def test_tensor_dtype_convert_half(self, device_type, dtype):
        v = np.random.uniform(size=1<<16) * 10
        if dtype == mp.kFloat64:
            # hmp only support convert float32 to half
            a = v.astype(to_np_dtype(dtype))
            xa = a.astype(np.float32)
            b = a.astype(np.float16)
            xb = xa.astype(np.float16)

            c = mp.from_numpy(a).to(device_type)
            d = c.to(mp.kHalf).cpu().numpy()
            diff = b - d
            ref_diff = b - xb

            assert((diff == ref_diff).all())
        else:
            a = v.astype(to_np_dtype(dtype))
            b = a.astype(np.float16)

            c = mp.from_numpy(a).to(device_type)
            d = c.to(mp.kHalf).cpu().numpy()
            assert((b == d).all())