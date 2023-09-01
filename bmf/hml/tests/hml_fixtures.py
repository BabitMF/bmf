#from typing_extensions import Required
import pytest
import bmf.lib._hmp as mp
import numpy as np
import os

DEFAULT_DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")
TEST_DATA_ROOT = os.environ.get('HMP_TEST_DATA_ROOT', DEFAULT_DATA_ROOT)

device_types = (mp.kCPU, mp.kCUDA)

scalar_types = (mp.kInt8, mp.kUInt8, mp.kInt16, mp.kUInt16, mp.kInt32,
                mp.kInt64, mp.kFloat32, mp.kFloat64, mp.kHalf)

numpy_dtypes = {
    mp.kInt8: np.int8,
    mp.kUInt8: np.uint8,
    mp.kInt16: np.int16,
    mp.kUInt16: np.uint16,
    mp.kInt32: np.int32,
    mp.kInt64: np.int64,
    mp.kFloat32: np.float32,
    mp.kFloat64: np.float64,
    mp.kHalf: np.float16
}


def to_np_dtype(dtype):
    return numpy_dtypes[dtype]


has_cuda = hasattr(mp, "cuda") and mp.device_count(mp.kCUDA) > 0

has_ffmpeg = mp.__config__.get('HMP_ENABLE_FFMPEG', 0)

has_opencv = mp.__config__.get('HMP_ENABLE_OPENCV', 0)


@pytest.fixture(params=[
    pytest.param(d, id=str(d)) for d in device_types if mp.device_count(d) > 0
])
def device_type(request):
    return request.param


@pytest.fixture
def with_half():
    return False


@pytest.fixture(params=[pytest.param(t, id=repr(t)) for t in scalar_types])
def dtype(request, with_half):
    if not with_half and request.param == mp.kHalf:
        pytest.skip("kHalf not support")
    else:
        return request.param


@pytest.fixture(params=[
    pytest.param(t, id=repr(t)) for t in (mp.kFloat32, mp.kFloat64, mp.kHalf)
])
def f_dtype(request, with_half):
    if not with_half and request.param == mp.kHalf:
        pytest.skip("kHalf not support")
    else:
        return request.param


@pytest.fixture(params=[
    pytest.param(t, id=repr(t))
    for t in (mp.kUInt8, mp.kUInt16, mp.kFloat32, mp.kHalf)
])
def img_dtype(request, with_half):
    if not with_half and request.param == mp.kHalf:
        pytest.skip("kHalf not support")
    else:
        return request.param


@pytest.fixture(
    params=[pytest.param(f, id=repr(f)) for f in (mp.kNCHW, mp.kNHWC)])
def channel_format(request):
    return request.param


@pytest.fixture(params=[pytest.param(f, id=repr(f)) for f in (1, 3, 4)])
def channels(request):
    return request.param


def get_data_file(fn):
    assert (TEST_DATA_ROOT is not None)
    fn = os.path.join(TEST_DATA_ROOT, fn)
    assert (os.path.exists(fn))
    return fn


@pytest.fixture()
def lenna_image():
    fn = get_data_file("common/Lenna_RGB24.yuv")
    width, height = 512, 512
    rgb = np.fromfile(fn, dtype=np.uint8).reshape((width, height, 3))
    rgba = np.pad(rgb, [(0, 0), (0, 0), (0, 1)],
                  mode='constant',
                  constant_values=1)
    return rgba
