import pytest
import bmf.lib._hmp as mp
from hml_fixtures import device_type, dtype
try:
    import torch
except ImportError:
    torch = None


@pytest.fixture
def with_half():
    return True


def test_torch_interop(device_type, dtype):
    if torch is None:
        pytest.skip("torch disabled")
    if dtype == mp.kUInt16:
        pytest.skip("torch not support")

    a = mp.arange(48, device=device_type).to(dtype).reshape((2, 6, 4))
    b = a.slice(1, 0, 6, 2)  # non-contiguous

    # to torch tensor
    c = b.torch()

    # convert back
    d = mp.from_torch(c)

    d.fill_(111)

    np_a = a.cpu().numpy()
    np_b = b.cpu().numpy()
    np_c = c.cpu().numpy()
    np_d = d.cpu().numpy()

    assert ((np_b == np_c).all())
    assert ((np_b == np_d).all())
    assert ((np_b == np_a[:, ::2, :]).all())
    assert ((np_b == 111).all())
