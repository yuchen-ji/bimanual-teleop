import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parents[2])]
from locked_grasp import FINGERS, LockedGrasp, distal_forces


class LockedGraspTests(unittest.TestCase):
    def state(self, suffix="tip_sensor_frame", opposed=True):
        return {"rail": [0, 0, 1, 0, 0, 0, 1],
                "rail_contacts": {f"l_{f}_{suffix}": .5 for f in FINGERS},
                "rail_contact_vectors": {f"l_{f}_{suffix}": [.5 if i == 0 or not opposed else -.5, 0, 0]
                                         for i, f in enumerate(FINGERS)}}

    def test_proximal_contacts_do_not_count_as_distal(self):
        self.assertEqual(sum(distal_forces(self.state("proximal")).values()), 0.)
        self.assertEqual(min(distal_forces(self.state("distal")).values()), .5)

    def test_attachment_requires_stable_opposition_and_clear_geometry(self):
        for suffix, opposed, intersects, expected in [("proximal", True, [], False),
                                                     ("tip_sensor_frame", False, [], False),
                                                     ("tip_sensor_frame", True, ["l_pinky_distal"], False),
                                                     ("tip_sensor_frame", True, [], True)]:
            with self.subTest(suffix=suffix, opposed=opposed, intersects=intersects), tempfile.TemporaryDirectory() as directory:
                source = SimpleNamespace(output=Path(directory), command=np.zeros(54), metrics={}, reset_index=0,
                                         runtime={}, task={}, pending_attachment=False, event=lambda *a, **k: None,
                                         finish=lambda *a: None)
                state = self.state(suffix, opposed)
                grasp = LockedGrasp(source, state)
                grasp.phase = "review"
                for _ in range(6):
                    grasp.step(state, .1)
                response = {"id": grasp.request_id, "ok": True, "geometry": {"intersecting_links": intersects},
                            "command": source.command.tolist()}
                (Path(directory)/"rpc/response.json").write_text(json.dumps(response))
                grasp.step(state, .1)
                self.assertEqual(source.pending_attachment, expected)

    def test_geometric_latch_needs_every_distal_surface_and_two_clear_reviews(self):
        with tempfile.TemporaryDirectory() as directory:
            source = SimpleNamespace(output=Path(directory), command=np.zeros(54), metrics={}, reset_index=0,
                                     runtime={}, task={"grasp_attachment_criterion": "distal_clearance"},
                                     pending_attachment=False, event=lambda *a, **k: None, finish=lambda *a: None)
            state = self.state(); state["rail_contacts"] = {}
            grasp = LockedGrasp(source, state)
            for tick, gap, expected in [(1, .001, False), (600000001, .00012, False), (1200000001, .00012, True)]:
                grasp.phase = "review"
                grasp.request("locked_grasp_check", state)
                geometry = {"time_ns": tick, "intersecting_links": [], "distal_gap_m": dict.fromkeys(FINGERS, .00012)}
                geometry["distal_gap_m"]["pinky"] = gap
                response = {"id": grasp.request_id, "ok": True, "geometry": geometry, "command": source.command.tolist()}
                (Path(directory)/"rpc/response.json").write_text(json.dumps(response))
                grasp.step(state, .1)
                self.assertEqual(source.pending_attachment, expected)

    def test_bottom_grasp_does_not_accept_close_side_contacts(self):
        with tempfile.TemporaryDirectory() as directory:
            source = SimpleNamespace(output=Path(directory), command=np.zeros(54), metrics={}, reset_index=0,
                                     runtime={}, task={"grasp_attachment_criterion": "distal_clearance", "grasp_contact_surface": "bottom"},
                                     pending_attachment=False, event=lambda *a, **k: None, finish=lambda *a: None)
            state = self.state(); state["rail_contacts"] = {}
            grasp = LockedGrasp(source, state)
            for tick, bottom, expected in [(1, False, False), (600000001, False, False),
                                           (1200000001, True, False), (1800000001, True, True)]:
                grasp.phase = "review"
                grasp.request("locked_grasp_check", state)
                geometry = {"time_ns": tick, "intersecting_links": [], "distal_gap_m": dict.fromkeys(FINGERS, .00012),
                            "bottom_contact_face_ok": bottom}
                response = {"id": grasp.request_id, "ok": True, "geometry": geometry, "command": source.command.tolist()}
                (Path(directory)/"rpc/response.json").write_text(json.dumps(response))
                grasp.step(state, .1)
                self.assertEqual(source.pending_attachment, expected)

    def test_final_review_rejects_finger_collision_and_lost_contact(self):
        for pair, gap in [([["l_index_finger_distal", "l_middle_finger_distal"]], .00012), ([], .0003)]:
            with self.subTest(pair=pair, gap=gap), tempfile.TemporaryDirectory() as directory:
                failures = []
                source = SimpleNamespace(output=Path(directory), command=np.zeros(54), metrics={}, reset_index=0,
                                         runtime={}, task={"grasp_attachment_criterion": "distal_clearance", "grasp_contact_surface": "bottom"},
                                         finish=lambda *args, failures=failures: failures.append(args))
                state = self.state(); state["rail_contacts"] = {}
                grasp = LockedGrasp(source, state)
                grasp.phase = "final_review"
                geometry = {"intersecting_links": [], "interfinger_intersections": pair,
                            "bottom_contact_face_ok": True, "distal_gap_m": dict.fromkeys(FINGERS, gap)}
                (Path(directory)/"rpc/response.json").write_text(json.dumps(
                    {"id": grasp.request_id, "ok": True, "geometry": geometry}))
                grasp.step(state, .1)
                self.assertEqual(failures[0][0], "failed")
                self.assertEqual(grasp.phase, "final_review")

    def test_translation_keeps_original_orientation_and_hand_open(self):
        with tempfile.TemporaryDirectory() as directory:
            source = SimpleNamespace(output=Path(directory), command=np.zeros(54), metrics={}, reset_index=0,
                                     runtime={}, event=lambda *a, **k: None, finish=lambda *a: None)
            state = self.state(); state["rail_contacts"] = {}
            grasp = LockedGrasp(source, state)
            grasp.plan = {"rail_xyz_m": [.01, .02, 1.01]}
            grasp.phase = "translate"
            for _ in range(40):
                np.testing.assert_array_equal(grasp.step(state, .1), np.zeros(54))
                np.testing.assert_array_equal(grasp.quaternion, [0, 0, 0, 1])
            np.testing.assert_allclose(grasp.rail_xyz, grasp.plan["rail_xyz_m"])
