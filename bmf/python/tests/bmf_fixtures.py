import bmf.hmp as mp

has_cuda = hasattr(mp, "cuda") and mp.device_count(mp.kCUDA) > 0

has_torch = mp.__config__.get("HMP_ENABLE_TORCH", 0)
