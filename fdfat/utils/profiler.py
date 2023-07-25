import time
import contextlib

class Profile(contextlib.ContextDecorator):
    # YOLOv5 Profile class. Usage: @Profile() decorator or 'with Profile():' context manager
    def __init__(self, name, max_sample=100):
        self.name = name
        self.samples = []
        self.max_sample=max_sample

    def __enter__(self):
        self.start = self.__time()
        return self

    def __exit__(self, type, value, traceback):
        t = self.__time() - self.start  # delta-time
        self.insert_time(t)

    def __time(self):
        return time.time()*1000

    def insert_time(self, value):
        self.dt = value
        self.samples.append(self.dt)
        if len(self.samples) > self.max_sample:
            self.samples.pop(0)

    def avg(self):
        if len(self.samples) == 0:
            return -1
        total = sum(self.samples)
        sizee = len(self.samples)
        target_t = int(total/sizee)
        return target_t

    def report(self):
        return self.avg()

    def print_report(self):
        print(f"[{int(time.time()*1000)}] {self.name}: {self.report()}ms")