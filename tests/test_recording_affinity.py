"""Recording workers stay unpinned so idle CPUs remain usable."""

import unittest
from unittest.mock import patch

from bimanual_teleop.common.affinity import apply_recording_affinity


class RecordingAffinityTests(unittest.TestCase):
    def test_recording_workers_are_not_pinned(self):
        with patch("os.sched_setaffinity") as setter:
            self.assertIsNone(apply_recording_affinity("background"))
            setter.assert_not_called()


if __name__ == "__main__":
    unittest.main()
