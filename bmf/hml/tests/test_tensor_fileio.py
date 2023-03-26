import pytest
from hml_fixtures import device_type, has_cuda, f_dtype, dtype, to_np_dtype, with_half
from hml_fixtures import mp, scalar_types, device_types
import numpy as np

class TestFileIO(object):
    def test_to_from_file(self):
        ref = mp.arange(65536, dtype=mp.float32)
        
        ref.tofile("_test.f32")

        d0 = mp.fromfile("_test.f32", dtype=mp.float32)
        d1 = mp.fromfile("_test.f32", dtype=mp.float32,
                         offset=1024, count=4096)
        
        r0 = ref.numpy()
        r1 = r0[1024:1024+4096]

        d0 = d0.numpy()
        d1 = d1.numpy()

        assert((r0 == d0).all())
        assert((r1 == d1).all())