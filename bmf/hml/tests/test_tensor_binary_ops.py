import pytest
from hml_fixtures import device_type, has_cuda, f_dtype, dtype, to_np_dtype, with_half
from hml_fixtures import mp, scalar_types, device_types
import numpy as np


class TestBinaryOps(object):
    """
    FIXME: type computation is not supported
    """

    @pytest.fixture
    def with_half(self):
        return True


    @pytest.fixture()
    def data(self, dtype):
        d = np.random.normal(size=1000).reshape((4, 5, 50)) * 10
        return d.astype(to_np_dtype(dtype))


    @pytest.fixture(params=["+", "-", "*", "/"])
    def arith_op(self, request):
        return request.param


    def test_mul(self, arith_op, data, device_type):
        data = data.copy()
        pdata = data[:, ::2, :]
        op = arith_op

        nop = op
        if op == '/':
            data[data==0] = 1

        scalar = data[0, 0, 0]
        expect0 = eval("data {} data".format(nop)).astype(data.dtype)
        expect1 = eval("data {} scalar".format(nop)).astype(data.dtype)
        expect2 = eval("expect0 {} data".format(nop)).astype(data.dtype)
        expect3 = eval("expect0 {} scalar".format(nop)).astype(data.dtype)
        expect4 = eval("scalar {} data".format(nop)).astype(data.dtype)
        expect5 = eval("pdata {} pdata".format(nop)).astype(data.dtype)

        data = mp.from_numpy(data).to(device_type)

        #as we cast scalar to the same type of tensor
        #this behaviour may change if type computation is enabled
        scalar = float(scalar) 

        d = data.clone()
        exec("self.c = d {} d".format(op))
        assert((self.c.cpu().numpy() == expect0).all())
        exec("self.c = d {} scalar".format(op))
        assert((self.c.cpu().numpy() == expect1).all())
        exec("self.c = scalar {} d".format(op))
        assert((self.c.cpu().numpy() == expect4).all())

        # inplace 
        self.d = d
        exec("self.d {}= d".format(op))
        assert((self.d.cpu().numpy() == expect0).all())
        exec("self.d {}= scalar".format(op))
        assert((self.d.cpu().numpy() == expect3).all())

        # non-contigous tensors
        d = data.clone().slice(1, 0, 5, 2)
        assert(d.is_contiguous == False)
        exec("self.c = d {} d".format(op))
        assert((self.c.cpu().numpy() == expect5).all())

