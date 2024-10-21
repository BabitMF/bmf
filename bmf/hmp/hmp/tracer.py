# Red
import bmf.hml.hmp as mp


def singleton(class_):
    instances = {}

    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]

    return getinstance


def get_device_type(device):
    device = str(device).lower()
    if 'cuda' in device:
        return mp.kCUDA
    else:
        assert ('cpu' in device)
        return mp.kCPU


class DummyTimer(object):

    def __init__(self, name, *args, **kwargs):
        self.name = name
        self.timer = 0
        self.counter = 0

    def __enter__(self, *args, **kwargs):
        self.old_timer = Tracer().current
        Tracer()._set_current(self)

    def __exit__(self, *args, **kwargs):
        Tracer()._set_current(self.old_timer)

    def __repr__(self):
        return "DummyTimer: {}".format(self.name)

    def elapsed(self):
        return 0


class Timer(object):

    def __init__(self, name, device='cpu'):
        self.device_type = get_device_type(device)
        self.timers = []
        self.name = name
        self.counter = 0

    def __enter__(self, *args, **kwargs):
        if self.counter >= len(self.timers):
            self.timers.append(mp.create_timer(self.device_type))
        self.timers[self.counter].start()

        self.old_timer = Tracer().current
        Tracer()._set_current(self)

    def __exit__(self, *args, **kwargs):
        self.timers[self.counter].stop()
        Tracer()._set_current(self.old_timer)
        self.counter += 1  # cleared by Tracer

    def __repr__(self):
        return "Timer<{}>".format(self.name)

    def elapsed(self):
        t = 0
        for timer in self.timers:
            t += timer.elapsed()
        return t


@singleton
class Tracer(object):

    def __init__(self):
        self.active = False
        self.current = self
        self.reset()

    def timer(self, name, device='cpu'):
        if self.current == self:
            full_name = name
        else:
            full_name = "{}.{}".format(self.current.name, name)

        if full_name not in self.timers:
            if self.active:
                self.timers[full_name] = Timer(full_name, device)
            else:
                self.timers[full_name] = DummyTimer(full_name)
            self.records[full_name] = []
        return self.timers.get(full_name)

    def _set_current(self, current):
        self.current = current

    def __enter__(self, *args, **kwargs):
        self.active = True
        for name, timer in self.timers.items():
            timer.counter = 0

    def __exit__(self, *args, **kwargs):
        self.active = False
        for name, timer in self.timers.items():
            if timer.counter:
                self.records[name].append(timer.elapsed())
            else:
                self.records[name].append(0)

    def reset(self):
        assert (self.active == False)
        self.timers = {}
        self.records = {}


def timer(name, device='cpu'):
    return Tracer().timer(name, device)


if __name__ == '__main__':
    import time
    tracer = Tracer()

    def f():
        with timer("test"):
            time.sleep(0.2)
            with timer("hello"):
                time.sleep(0.1)
                for i in range(10):
                    with timer("hello"):
                        time.sleep(0.01)

        #with timer("test"): #assert Failed()
        with timer("hello"):
            time.sleep(0.5)

    for i in range(10):
        with tracer:
            with timer("top"):
                f()

    print(tracer.timers)
    print(tracer.records)
