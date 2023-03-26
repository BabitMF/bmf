import hmp as mp

has_cuda = hasattr(mp, "cuda") and mp.device_count(mp.kCUDA) > 0