import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parents[2])]

import test_preinsert_sim
from four_fingers import FINGERS, FourFingerClosure


class FourFingerTests(unittest.TestCase):
    @staticmethod
    def clear():
        return {"intersecting_links": [], "finger_gap_m": dict.fromkeys(FINGERS, .02)}

    def test_only_flexion_moves_and_individual_contact_stops_advancement(self):
        command = np.arange(54) * .001
        closure = FourFingerClosure(command, np.ones(20), 3.)
        np.testing.assert_array_equal(closure.step({}, .1), command)
        closure.review(self.clear())
        frozen = closure.step({}, .1)
        command_before = frozen.copy()
        for _ in range(120):
            frozen = closure.step({"rail_contacts": {"l_index_finger_tip": 1.}}, 1/60)
            if closure.phase == "check":
                closure.review(self.clear())
        np.testing.assert_array_equal(frozen[closure.indices[0]], command_before[closure.indices[0]])
        unaffected = np.setdiff1d(np.arange(54), np.concatenate(closure.indices))
        np.testing.assert_array_equal(frozen[unaffected], command[unaffected])
        self.assertGreater(frozen[22], command_before[22])
        for _ in range(120):
            resumed = closure.step({}, 1/60)
            if closure.phase == "check":
                closure.review(self.clear())
        self.assertGreater(resumed[18], frozen[18])

    def test_intersection_retreats_only_affected_finger_and_reduces_step(self):
        closure = FourFingerClosure(np.zeros(54), np.ones(20), 3.)
        closure.review(self.clear())
        closure.step({}, .35)
        closure.step({}, .6)
        others = closure.command[22:34].copy()
        closure.review({**self.clear(), "intersecting_links": ["l_index_finger_tip"]})
        self.assertEqual(closure.phase, "retreat")
        closure.review(self.clear())  # Stale response cannot cancel the retreat.
        self.assertEqual(closure.phase, "retreat")
        closure.step({}, .35)
        np.testing.assert_allclose(closure.command[closure.indices[0]], 0)
        np.testing.assert_allclose(closure.command[22:34], others)
        self.assertAlmostEqual(closure.step_rad[0], np.radians(.5))

    def test_initial_intersection_prevents_any_closure(self):
        closure = FourFingerClosure(np.zeros(54), np.ones(20), 3.)
        closure.review({**self.clear(), "intersecting_links": ["l_wrist"]})
        self.assertIn("not started", closure.failure)
        np.testing.assert_array_equal(closure.command, 0)

    def test_force_overload_rolls_back_last_increment(self):
        closure = FourFingerClosure(np.zeros(54), np.ones(20), 3.)
        closure.review(self.clear())
        closure.step({}, .35)
        closure.step({"rail_contacts": {"l_index_finger_tip": 4.}}, .01)
        self.assertEqual(closure.phase, "retreat")
        closure.step({}, .35)
        np.testing.assert_allclose(closure.command[closure.indices[0]], 0)

    def run_closure(self, contacts):
        with tempfile.TemporaryDirectory() as directory:
            source = test_preinsert_sim.ControllerTests().controller(directory, headless=False)
            source.grasp_mode = "physics"
            source.task["until"] = "four_fingers"
            source.reset()
            initial = source.command.copy()
            state = {"q": initial.tolist(), "rail_contacts": contacts, "time_ns": 1, "tick": 1}
            previous = None
            for _ in range(10000):
                state["q"] = source.step(state, 1/60).tolist()
                path = Path(directory) / "rpc/request.json"
                if path.exists():
                    request = json.loads(path.read_text())
                    if request["id"] != previous:
                        previous = request["id"]
                        self.assertEqual(request["stage"], "four_fingers_check")
                        (path.parent / "response.json").write_text(json.dumps({
                            "id": previous, "ok": True, "geometry": self.clear()}))
                source.observe(state, [np.zeros((8, 8, 3), dtype=np.uint8)]*3)
                self.assertTrue(source.support_active)
                self.assertFalse(source.pending_placement or source.pending_attachment)
                if source.finished:
                    break
            self.assertTrue(source.finished)
            self.assertTrue(source.pause_physics_requested)
            self.assertIsNone(source.request_id)  # No perception or placement RPC.
            np.testing.assert_array_equal(source.command[14:18], initial[14:18])
            np.testing.assert_array_equal(source.command[:14], initial[:14])
            result = json.loads((Path(directory) / "result.json").read_text())
            self.assertFalse(result["physical_grasp_validated"])
            self.assertFalse(result["insertion_executed"])
            self.assertTrue((source.trial_directory / "four_fingers_pose.json").exists())
            source.close()
            return result

    def test_supported_contact_is_distinct_from_unsupported_grasp(self):
        result = self.run_closure({"l_"+f+"_tip": 1. for f in FINGERS})
        self.assertEqual(result["status"], "supported_four_finger_contact")

    def test_missing_contacts_produce_preview_not_grasp_success(self):
        result = self.run_closure({})
        self.assertEqual(result["status"], "four_finger_preview_complete")
        self.assertEqual(result["metrics"]["four_fingers_missing_contact"], list(FINGERS))


if __name__ == "__main__":
    unittest.main()
