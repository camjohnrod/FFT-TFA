# Wall-clock time per named phase of an online solve, accumulated per load step. Timed calls must not nest.

import functools
import time

online_time_groups = ("material_update", "induced_strain", "reference_sensitivity", "reference_inverse",
                      "correction_solve")

class OnlineTimer:
    def __init__(self):
        self.open_group = None
        self.reset()

    def reset(self):
        self.time_per_group = [0.0] * len(online_time_groups)
        self.calls_per_group = [0] * len(online_time_groups)

online_timer = OnlineTimer()

def timed_online(group):
    group_id = online_time_groups.index(group)
    def decorator(function):
        @functools.wraps(function)
        def timed_function(*args, **kwargs):
            if online_timer.open_group is not None:
                raise RuntimeError(f"online time group {group!r} started inside {online_timer.open_group!r}, "
                                   "so its time would be counted twice.")
            online_timer.open_group = group
            start_time = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                online_timer.time_per_group[group_id] += time.perf_counter() - start_time
                online_timer.calls_per_group[group_id] += 1
                online_timer.open_group = None
        return timed_function
    return decorator

def get_online_timer_overhead_per_call(call_count=100_000):
    def untimed_function():
        pass
    timed_function = timed_online(online_time_groups[0])(untimed_function)

    start_time = time.perf_counter()
    for _ in range(call_count):
        timed_function()
    timed_duration = time.perf_counter() - start_time
    start_time = time.perf_counter()
    for _ in range(call_count):
        untimed_function()
    untimed_duration = time.perf_counter() - start_time
    online_timer.reset()
    return max(timed_duration - untimed_duration, 0) / call_count
