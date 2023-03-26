
import pytest
from hml_fixtures import device_type, has_cuda, f_dtype, dtype, to_np_dtype, with_half
from hml_fixtures import mp, scalar_types, device_types
import numpy as np



class TestUnaryOps(object):
    @pytest.fixture
    def with_half(self):
        return True

    @pytest.fixture
    def round_data(self):
        data = np.random.uniform(size=8192) * np.finfo(np.float32).max
        data[0] = np.inf
        data[1] = -np.inf
        return data


    def test_round(self, round_data, device_type, f_dtype):
        data = round_data.astype(to_np_dtype(f_dtype))
        expect = np.round(data)

        data = mp.from_numpy(data).to(device_type)
        actual0 = data.round()
        actual1 = data.round_()

        assert((actual0.cpu().numpy() == expect).all())
        assert((actual1.cpu().numpy() == expect).all())


    def test_ceil(self, round_data, device_type, f_dtype):
        data = round_data.astype(to_np_dtype(f_dtype))
        expect = np.ceil(data)

        data = mp.from_numpy(data).to(device_type)
        actual0 = data.ceil()
        actual1 = data.ceil_()

        assert((actual0.cpu().numpy() == expect).all())
        assert((actual1.cpu().numpy() == expect).all())


    def test_floor(self, round_data, device_type, f_dtype):
        data = round_data.astype(to_np_dtype(f_dtype))
        expect = np.floor(data)

        data = mp.from_numpy(data).to(device_type)
        actual0 = data.floor()
        actual1 = data.floor_()

        assert((actual0.cpu().numpy() == expect).all())
        assert((actual1.cpu().numpy() == expect).all())
        

    def test_abs(self, device_type, dtype):
        if dtype in [mp.kFloat32, mp.kFloat64, mp.kHalf]:
            info = np.finfo(to_np_dtype(dtype))
            data = np.random.uniform(size=1024)
            data[0] = np.inf
            data[1] = -np.inf 
        else:
            info = np.iinfo(to_np_dtype(dtype))
            data = np.random.uniform(size=1024) * 1000
        data = data.astype(to_np_dtype(dtype))

        expect = np.abs(data)

        data = mp.from_numpy(data).to(device_type)
        actual0 = data.abs()
        actual1 = data.abs_() 
        assert((actual0.cpu().numpy() == expect).all())
        assert((actual1.cpu().numpy() == expect).all())


    def test_clip(self, device_type, dtype):
        data = np.arange(-100, 100)
        data = data.astype(to_np_dtype(dtype))

        expect0 = np.clip(data, 5, 10)
        expect1 = np.clip(data, 11, 11)

        data = mp.from_numpy(data).to(device_type)
        actual0 = data.clip(5, 10)
        actual1 = data.clip(11, 11)
        assert((actual0.cpu().numpy() == expect0).all())
        assert((actual1.cpu().numpy() == expect1).all())

        #
        with pytest.raises(RuntimeError):
            data.clip(10, 0)

