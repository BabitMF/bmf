
import pytest
from hml_fixtures import device_type, has_cuda
from hml_fixtures import mp


@pytest.mark.skipif(not has_cuda, reason="need cuda device")
def test_event():
    stream = mp.create_stream(mp.kCUDA)
    cpu_tensor = mp.empty((128<<20,), device=mp.kCPU)
    dev_tensor = mp.empty_like(cpu_tensor, device=mp.kCUDA)

    event = mp.cuda.Event(False)
    assert(event.is_created() == False)

    with stream:
        mp.copy(dev_tensor, cpu_tensor)
        event.record()
        assert(event.is_created() == True)
        assert(event.query() == False)
    
    event.synchronize()
    assert(event.query() == True)
