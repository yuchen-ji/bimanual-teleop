"""Recording workers are not CPU-pinned.

Motion control keeps the process affinity. Restricting that process to two
physical cores made the host watchdog miss valid targets. Pinning only the
recording workers onto the other cores then left idle CPUs unreachable, so
the frame rings overflowed while the machine still had spare cores. Those
workers use a lower nice value instead.
"""


def apply_recording_affinity(role):
    """Leave the caller on its current CPUs. ``role`` is accepted for callers."""
    del role
    return None
