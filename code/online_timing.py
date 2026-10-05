# Wall-clock time per named phase of an online solve, accumulated per load step. Timed calls may nest: each group is
# charged only its own time, with the time of every timed call made inside it charged to that call's group instead.
# So the P products inside a Newton step's Krylov solve count as induced strain, not as correction solve, and the
# groups never count the same time twice.

import functools
import time

online_time_groups = ("material_update", "induced_strain", "sensitivity", "reference_inverse", "correction_solve")

class OnlineTimer:
    def __init__(self):
        # One [group index, start time, time of the timed calls nested inside it] per timed call in progress,
        # innermost last.
        self.open_calls = []
        self.reset()

    def reset(self):
        if self.open_calls:
            raise RuntimeError("the online timer was reset inside a timed call.")
        self.time_per_group = [0.0] * len(online_time_groups)

    def start(self, group_id):
        self.open_calls.append([group_id, time.perf_counter(), 0.0])

    def stop(self):
        group_id, start_time, nested_time = self.open_calls.pop()
        elapsed_time = time.perf_counter() - start_time
        self.time_per_group[group_id] += elapsed_time - nested_time
        if self.open_calls:
            self.open_calls[-1][2] += elapsed_time

online_timer = OnlineTimer()

def timed_online(group):
    group_id = online_time_groups.index(group)
    def decorator(function):
        @functools.wraps(function)
        def timed_function(*args, **kwargs):
            online_timer.start(group_id)
            try:
                return function(*args, **kwargs)
            finally:
                online_timer.stop()
        return timed_function
    return decorator
