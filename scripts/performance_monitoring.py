import logging
import time
import tracemalloc
import linecache
import os


class Timer:
    def __init__(self):
        self.start = time.time()
        self.steps = [self.start]

    def launch(self):
        t = time.time()
        self.start = t
        self.steps = [t]
        t = self.printable_time(t)
        return t

    def step(self):
        t = time.time()
        self.steps.append(t)
        t = self.printable_time(t)
        return t

    def total_duration(self):
        now = time.time()
        d = self.printable_time(now - self.start)
        return d

    def last_step_duration(self):
        try:
            d = self.steps[-1] - self.steps[-2]
            d = self.printable_time(d)
            return d
        except Exception:
            raise Exception('No existing step to compare with')

    def printable_time(self, time):
        return ('%.2fs' % time).lstrip('0')



class MemoryMonitoring:
    def __init__(self):
        tracemalloc.start()
        self.snapshots = []

    def take_snapshot(self):
        snapshot = tracemalloc.take_snapshot()
        self.snapshots.append(snapshot)

    def display_top(self, snapshot, key_type='lineno', limit=5):
        print ("snapshot printing strats")
        snapshot = snapshot.filter_traces((
            tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
            tracemalloc.Filter(False, "<unknown>"),
        ))
        top_stats = snapshot.statistics(key_type)

        print("Top %s lines" % limit)
        for index, stat in enumerate(top_stats[:limit], 1):
            frame = stat.traceback[0]
            # replace "/path/to/module/file.py" with "module/file.py"
            filename = os.sep.join(frame.filename.split(os.sep)[-2:])
            print("#%s: %s:%s: %.1f KiB"
                  % (index, filename, frame.lineno, stat.size / 1024))
            line = linecache.getline(frame.filename, frame.lineno).strip()
            if line:
                print('    %s' % line)

        other = top_stats[limit:]
        if other:
            size = sum(stat.size for stat in other)
            print("%s other: %.1f KiB" % (len(other), size / 1024))
        total = sum(stat.size for stat in top_stats)
        print("Total allocated size: %.1f KiB" % (total / 1024))


    def current_memory_state(self, limit=5):
        self.take_snapshot()
        self.display_top(self.snapshots[-1], limit=5)

class PerfLogger:
    def __init__(self, timer = None, memorymonitoring = None):
        if isinstance(timer, Timer):
            self.timer = timer
        elif timer is None :
            self.timer = Timer()
        else:
            logging.error("Wrong type of value setted for timer")

        if isinstance(memorymonitoring, MemoryMonitoring):
            self.mm = memorymonitoring
        elif memorymonitoring is None :
            self.mm = MemoryMonitoring()
        else:
            logging.error("Wrong type of value setted for memory monitoring")

    def log_step(self):
        self.timer.step()
        print(str(self.timer.last_step_duration()))
        self.mm.current_memory_state()