import time
# decorator to time a function call for metrics
def timer(func):
    def wrapper(self, *args, **kwargs):
        start = time.time()
        result = func(self, *args, **kwargs)
        inference_time = time.time() - start
        return result, inference_time
    return wrapper
