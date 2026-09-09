import time

TASK_QUEUED = 0
TASK_PREPARING = 1
TASK_EXECUTING = 2

_TASK_START_TIMES = None
_TASK_PHASES = None
_CURRENT_TASK_INDEX = None


def init_task_start_times(task_start_times, task_phases):
    global _TASK_START_TIMES, _TASK_PHASES, _CURRENT_TASK_INDEX
    _TASK_START_TIMES = task_start_times
    _TASK_PHASES = task_phases
    _CURRENT_TASK_INDEX = None


def run_with_start_tracking(task_index, func, args):
    if _TASK_START_TIMES is None:
        raise RuntimeError("Task start-time storage is not initialized")
    global _CURRENT_TASK_INDEX
    _CURRENT_TASK_INDEX = task_index
    _TASK_PHASES[task_index] = TASK_PREPARING
    _TASK_START_TIMES[task_index] = time.monotonic()
    return func(*args)


def mark_task_execution_start():
    if (
        _TASK_START_TIMES is None
        or _TASK_PHASES is None
        or _CURRENT_TASK_INDEX is None
    ):
        return
    task_index = _CURRENT_TASK_INDEX
    if _TASK_PHASES[task_index] != TASK_PREPARING:
        return
    _TASK_PHASES[task_index] = TASK_EXECUTING
    _TASK_START_TIMES[task_index] = time.monotonic()


def elapsed_since_task_start(task_start_times, task_index, now=None):
    started_at = task_start_times[task_index]
    if started_at == 0:
        return None
    current_time = time.monotonic() if now is None else now
    return current_time - started_at


def reset_task_start_times(task_start_times, task_indices):
    """Mark tasks as queued again, so a resubmit is not judged against the
    timestamp its previous attempt left behind."""
    for k in task_indices:
        task_start_times[k] = 0


def timeout_for_phase(task_phases, task_index, timeout, build_timeout):
    if task_phases[task_index] == TASK_PREPARING:
        return build_timeout
    return timeout
