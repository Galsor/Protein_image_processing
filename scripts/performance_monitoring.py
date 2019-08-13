import time
import tracemalloc
import linecache
import os


class Timer:
    def __init__(self):
        self.start = time.time()

    def restart(self):
        self.start = time.time()

    def duration(self):
        now = time.time()
        return now - self.start

    def stop(self):
        del self


class MemoryMonitoring:
    def __init__(self):
        tracemalloc.start()
        self.snapshots = []

    def take_snapshot(self):
        snapshot = tracemalloc.take_snapshot()
        self.snapshots.append(snapshot)

    def display_top(self, snapshot, key_type='lineno', limit=5):
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