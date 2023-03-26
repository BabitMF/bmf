
import pytest
from hml_fixtures import device_type, has_cuda
from hml_fixtures import mp


def test_device_create(device_type):
    # default
    d0 = mp.Device()
    assert(d0.type() == mp.kCPU)
    assert(d0.index() == 0)

    #
    dev_str = 'cpu'
    if device_type == mp.kCUDA:
        dev_str = 'cuda'
    elif device_type != mp.kCPU:
        assert(False) #unsupport device

    #index out of range
    with pytest.raises(RuntimeError):
        mp.Device("{}:{}".format(dev_str, mp.device_count(device_type)))

    # invalid device index
    with pytest.raises(RuntimeError):
        mp.Device("{}:".format(dev_str))
    with pytest.raises(RuntimeError):
        mp.Device("{}:+".format(dev_str))

    d1 = mp.Device(device_type)
    d2 = mp.Device(device_type, 0)
    d3 = mp.Device(dev_str + ":0")
    d4 = mp.Device(dev_str)
    devs = [d1, d2, d3, d4]
    for d in devs:
        assert(d.type() == device_type)
        assert(d.index() == 0)
        assert(d == d1)


def test_device_guard():
    assert(mp.device_count(mp.kCPU) == 1)
    assert(mp.current_device(mp.kCPU).type() == mp.kCPU)


@pytest.mark.skipif(not has_cuda, reason="request cuda devices")
def test_cuda_device_guard():
    cuda0 = mp.Device(mp.kCUDA)
    assert(mp.current_device(mp.kCUDA) == cuda0)

    #
    if mp.device_count(mp.kCUDA) > 1:
        cuda1 = mp.Device("cuda:1")
        with cuda1:
            assert(mp.current_device(mp.kCUDA) == cuda1)
        assert(mp.current_device(mp.kCUDA) == cuda0)

