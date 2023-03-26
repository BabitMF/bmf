
import pytest
from hml_fixtures import device_type, has_cuda
from hml_fixtures import mp
import time


def test_timer(device_type):
    cpu_tensor = mp.empty((64<<20,), device=mp.kCPU)
    dev_tensor = mp.empty_like(cpu_tensor, device=device_type)

    stream = mp.create_stream(device_type)
    timer = mp.create_timer(device_type)
    stime = time.time()
    with stream:
        timer.start()
        mp.copy(dev_tensor, cpu_tensor)
        timer.stop()
    stream.synchronize()
    elapsed = time.time() - stime

    diff = abs(elapsed - timer.elapsed())
    assert(diff < elapsed*0.01) #??


@pytest.mark.skipif(not has_cuda, reason="need cuda device")
def test_cuda_timer():
    timer = mp.create_timer(mp.kCUDA)
    stream = mp.create_stream(mp.kCUDA)

    with pytest.raises(RuntimeError):
        timer.stop()

    with pytest.raises(RuntimeError):
        timer.elapsed()

    cpu_tensor = mp.empty((128<<20,), device=mp.kCPU, pinned_memory=True)
    dev_tensor = mp.empty_like(cpu_tensor, device=mp.kCUDA)
    with stream:
        timer.start()
        mp.copy(dev_tensor, cpu_tensor)
        timer.stop()

        print(stream.query())
        #not finished
        with pytest.raises(RuntimeError):
            print(timer.elapsed())
    
    stream.synchronize()
    print(timer.elapsed())


