
import pytest
from hml_fixtures import has_cuda
from hml_fixtures import mp


@pytest.mark.skipif(not has_cuda, reason="need cuda devices")
class TestCUDAStream(object):
    @pytest.fixture
    def stream(self):
        return mp.create_stream(mp.kCUDA)


    @pytest.fixture
    def stream1(self):
        return mp.create_stream(mp.kCUDA)


    def test_stream_guard(self, stream, stream1):
        default_stream = mp.current_stream(mp.kCUDA)
        assert(default_stream.handle() == 0)

        with stream:
            assert(mp.current_stream(mp.kCUDA) == stream)
            with stream1:
                assert(mp.current_stream(mp.kCUDA) == stream1)
            assert(mp.current_stream(mp.kCUDA) == stream)
        assert(mp.current_stream(mp.kCUDA) == default_stream)
        

    def test_async_call(self, stream):
        cpu_tensor = mp.empty((4<<20,), device='cpu', pinned_memory=True)
        cuda_tensor = mp.empty_like(cpu_tensor, device='cuda:0')
        with stream:
            mp.copy(cuda_tensor, cpu_tensor)
            assert(stream.query() == False)
        stream.synchronize()
        assert(stream.query() == True)


# exprimental
def test_cpu_stream():
    default_stream = mp.current_stream(mp.kCPU)
    stream = mp.create_stream(mp.kCPU)
    assert(stream == default_stream)