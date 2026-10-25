"""Hardware-free checks for isolated keyboard jog and Hand2 homing."""

from contextlib import redirect_stderr
from itertools import count
import io
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import time
import unittest
from unittest.mock import Mock, call, patch

import yaml

from bimanual_teleop.devices.wuji.adapter import JOINT_NAMES
from bimanual_teleop.devices.tianji.model import DEFAULT_MODEL
from bimanual_teleop.control.arm import jog as tianji_jog
from bimanual_teleop.control.hand import home as wuji_home
from bimanual_teleop.types import (ControlProfile, JointState, Pose, Sample,
                                   SampleHeader, SampleRef, Submission)


POSE = Pose("left_base", "left_flange", (0., 0., 0.), (0., 0., 0., 1.))


class _Keys:
    def __init__(self, *keys):
        self.keys = iter(keys)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def read(self, timeout):
        return next(self.keys, "q")


class TianjiJogTests(unittest.TestCase):
    def setUp(self):
        for name in ("NonblockingTerminal", "confirm_motion"):
            options = {"return_value": True} if name == "confirm_motion" else {}
            patcher = patch.object(tianji_jog, name, **options)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_jog_uses_shared_config_without_manual_ip(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.yaml"
            settings = tianji_jog.load_config()
            settings["controller_ip"] = "192.0.2.8"
            config.write_text(yaml.safe_dump(settings))
            driver = SimpleNamespace(start=lambda: None, close=lambda: None)
            with patch.object(tianji_jog, "TianjiDriver", return_value=driver) as factory, \
                 patch.object(tianji_jog, "TianjiKinematics"), \
                 patch.object(tianji_jog, "prepare_initial_pose") as prepare, \
                 patch.object(tianji_jog, "run_jog"), redirect_stderr(io.StringIO()):
                self.assertEqual(tianji_jog.main(["--side", "left", "--config", str(config)]), 0)
            self.assertEqual(factory.call_args.args[0], "192.0.2.8")
            prepare.assert_called_once()

    def test_motion_prepares_selected_arm_before_creating_jog_devices(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, model, library = root / "custom.yaml", root / "model.MvKDCfg", root / "custom.so"
            settings = tianji_jog.load_config()
            config.write_text(yaml.safe_dump(settings))
            model.write_bytes(DEFAULT_MODEL.read_bytes())
            events = []
            terminal = _Keys(None)
            driver = SimpleNamespace(start=lambda: events.append("start"), close=lambda: events.append("close"))
            with patch.object(tianji_jog, "NonblockingTerminal", return_value=terminal), \
                 patch.object(tianji_jog, "prepare_initial_pose",
                              side_effect=lambda **kwargs: events.append("prepare")) as prepare, \
                 patch.object(tianji_jog, "TianjiKinematics",
                              side_effect=lambda *args: events.append("kinematics")) as kinematics, \
                 patch.object(tianji_jog, "TianjiDriver",
                              side_effect=lambda *args, **kwargs: events.append("driver") or driver) as factory, \
                 patch.object(tianji_jog, "run_jog",
                              side_effect=lambda *args, **kwargs: events.append("jog")) as run, \
                 redirect_stderr(io.StringIO()):
                self.assertEqual(tianji_jog.main([
                    "--side", "right", "--tianji-config", str(config),
                    "--ip", "192.0.2.9", "--sdk-root", str(library), "--model", str(model)]), 0)
            self.assertEqual(events, ["prepare", "kinematics", "driver", "start", "jog", "close"])
            prepare.assert_called_once_with(config=config, ip="192.0.2.9", side="right",
                                            sdk_root=library, model=model, terminal=terminal)
            factory.assert_called_once_with("192.0.2.9", library, model_path=model)
            kinematics.assert_called_once_with(library, model)
            self.assertEqual(run.call_args.kwargs["side"], "right")

    def test_preparation_failure_or_cancel_prevents_jog_connection(self):
        for reason in ("初始位姿准备失败", "初始位姿准备已取消"):
            with self.subTest(reason=reason), \
                 patch.object(tianji_jog, "NonblockingTerminal", return_value=_Keys(None)), \
                 patch.object(tianji_jog, "prepare_initial_pose", side_effect=RuntimeError(reason)), \
                 patch.object(tianji_jog, "TianjiKinematics") as kinematics, \
                 patch.object(tianji_jog, "TianjiDriver") as driver, \
                 patch.object(tianji_jog, "run_jog") as run, redirect_stderr(io.StringIO()):
                self.assertEqual(tianji_jog.main(["--side", "left"]), 1)
            kinematics.assert_not_called()
            driver.assert_not_called()
            run.assert_not_called()

    def test_key_directions_and_base_rotation(self):
        self.assertEqual(tianji_jog.jog_pose(POSE, "w").position_m, (.005, 0., 0.))
        self.assertEqual(tianji_jog.jog_pose(POSE, "s").position_m, (-.005, 0., 0.))
        self.assertEqual(tianji_jog.jog_pose(POSE, "a").position_m, (0., .005, 0.))
        self.assertEqual(tianji_jog.jog_pose(POSE, "r").position_m, (0., 0., .005))
        rotation = tianji_jog.jog_pose(POSE, "u").orientation_xyzw
        self.assertAlmostEqual(rotation[2], math.sin(math.radians(1)))
        self.assertAlmostEqual(rotation[3], math.cos(math.radians(1)))

    def test_planner_rejects_ik_limit_without_advancing_goal(self):
        class Limits:
            def ik(self, side, candidate, reference):
                if candidate.position_m[0] > 0:
                    raise RuntimeError("joint limit")

        planner = tianji_jog.JogPlanner(POSE)
        with self.assertRaisesRegex(RuntimeError, "joint limit"):
            planner.step("w", time.monotonic_ns(), side="left", kinematics=Limits(),
                         reference_rad=(0.,)*7, translation_m=.005,
                         rotation_rad=math.radians(2))
        self.assertEqual(planner.goal, POSE)

    def test_unselected_arm_does_not_block_readonly_feedback(self):
        now = time.monotonic_ns()
        bad = SimpleNamespace(source_sequence=1, error=4, joints=JointState((), (None,)*7))
        good = SimpleNamespace(source_sequence=2, error=0,
                               joints=JointState((), (0.,)*7))
        sample = Sample(SampleHeader(SampleRef("tianji", "e", 1), now, False),
                        SimpleNamespace(arms={"left": good, "right": bad}))
        tracker = tianji_jog.SideFeedbackTracker("left", 50_000_000)
        self.assertEqual(tracker.observe(sample, now), (0.,)*7)
        self.assertIsNone(tracker.observe(sample, now + 50_000_001))

    def test_ik_failure_holds_engaged_arm(self):
        now = time.monotonic_ns()
        state = SimpleNamespace(source_sequence=1, error=0, joints=JointState((), (0.,)*7))
        sample = Sample(SampleHeader(SampleRef("tianji", "e", 1), now, True),
                        SimpleNamespace(arms={"left": state, "right": state}))
        driver = SimpleNamespace(watchdog_ns=50_000_000, get_latest=lambda: sample)
        calls = []

        class Executor:
            def __init__(self, driver, kinematics):
                self.engagement_poses = {"left": POSE}
                self.engagement_ref = sample.header.ref

            def configure(self, profile):
                calls.append("configure")

            def engage(self):
                calls.append("engage")

            def submit(self, target):
                calls.append("submit")
                return Submission(target.command_id, True)

            def request_hold(self, reason):
                calls.append(("hold", reason))

        class Limits:
            def ik(self, side, pose, reference):
                if pose.position_m[0] > 0:
                    raise RuntimeError("joint limit")

        with patch.object(tianji_jog, "TianjiCartesianExecutor", Executor):
            tianji_jog.run_jog(driver, Limits(), ControlProfile("p", "cartesian_impedance", {}),
                               side="left", translation_m=.005,
                               rotation_rad=math.radians(2), transition_s=.25,
                               terminal=_Keys("\r", "w", "q"), emit=lambda _: None)
        self.assertEqual(calls[:3], ["configure", "engage", "submit"])
        self.assertTrue(any(item[0] == "hold" and "joint limit" in item[1]
                            for item in calls if isinstance(item, tuple)))

    def test_calibration_defers_configuration_until_enter(self):
        for keys, expected in ((_Keys("q"), []), (_Keys("\r", "q"), ["configure", "engage", "submit", "hold"])):
            calls = []
            executor = SimpleNamespace(
                configure=lambda profile: calls.append("configure"),
                engage=lambda: calls.append("engage"),
                engagement_poses={"left": POSE}, engagement_ref=None,
                submit=lambda target: calls.append("submit") or Submission(target.command_id, True),
                request_hold=lambda reason: calls.append("hold"))
            driver = SimpleNamespace(watchdog_ns=50_000_000, get_latest=lambda: None)
            with patch.object(tianji_jog, "TianjiCartesianExecutor", return_value=executor):
                tianji_jog.run_jog(driver, None, ControlProfile("p", "cartesian_impedance", {}),
                                   side="left", translation_m=.005, rotation_rad=.02, transition_s=.25,
                                   terminal=keys, emit=lambda _: None, configure_on_engage=True)
            self.assertEqual(calls, expected)

    def test_position_mode_persists_across_keyboard_pause_and_resume(self):
        for options, engagement in (({}, call()), ({"control_mode": "position"}, call(control_mode="position"))):
            with self.subTest(options=options):
                now = time.monotonic_ns()
                arm = SimpleNamespace(source_sequence=1, error=0, joints=JointState((), (0.,)*7))
                sample = Sample(SampleHeader(SampleRef("tianji", "e", 1), now, True),
                                SimpleNamespace(arms={"left": arm}))
                driver = SimpleNamespace(watchdog_ns=50_000_000, get_latest=lambda: sample)
                executor = Mock(engagement_poses={"left": POSE}, engagement_ref=sample.header.ref)
                executor.submit.side_effect = lambda target: Submission(target.command_id, True)
                kinematics = SimpleNamespace(ik=lambda *args: (0.,)*7)
                with patch.object(tianji_jog, "TianjiCartesianExecutor", return_value=executor), \
                        patch.object(tianji_jog.time, "monotonic_ns", side_effect=count(now, 1_000_000)), \
                        patch.object(tianji_jog.time, "sleep"):
                    tianji_jog.run_jog(driver, kinematics, ControlProfile("p", "cartesian_impedance", {}),
                                       side="left", translation_m=.005, rotation_rad=.02, transition_s=.005,
                                       terminal=_Keys("\r", "w", "i", " ", "\r", "w", "q"),
                                       emit=lambda _: None, configure_on_engage=True, **options)
                self.assertEqual(executor.engage.call_args_list, [engagement, engagement])
                executor.configure.assert_called_once()
                self.assertEqual(executor.submit.call_count, 5)
                targets = [item.args[0].tool_poses["left"] for item in executor.submit.call_args_list]
                self.assertTrue(any(pose.position_m[0] > 0 for pose in targets))
                self.assertTrue(any(pose.orientation_xyzw[0] > 0 for pose in targets))
                self.assertEqual(executor.request_hold.call_args_list,
                                 [call("keyboard pause"), call("keyboard jog exited")])

    def test_invalid_control_mode_cannot_configure_or_engage(self):
        with patch.object(tianji_jog, "TianjiCartesianExecutor") as factory:
            with self.assertRaisesRegex(ValueError, "control_mode"):
                tianji_jog.run_jog(None, None, ControlProfile("p", "cartesian_impedance", {}),
                                   side="left", translation_m=.005, rotation_rad=.02, transition_s=.25,
                                   control_mode="pvt")
            factory.assert_not_called()

    def test_jog_after_engagement_ignores_unselected_fault(self):
        now = time.monotonic_ns()
        good = SimpleNamespace(source_sequence=1, error=0, joints=JointState((), (0.,)*7))
        bad = SimpleNamespace(source_sequence=1, error=42, joints=JointState((), (None,)*7))
        sample = Sample(SampleHeader(SampleRef("tianji", "e", 1), now, False),
                        SimpleNamespace(arms={"left": good, "right": bad}))
        driver = SimpleNamespace(watchdog_ns=50_000_000, get_latest=lambda: sample)
        calls = []

        class Executor:
            def __init__(self, driver, kinematics):
                self.engagement_poses = {"left": POSE}
                self.engagement_ref = sample.header.ref

            def configure(self, profile):
                pass

            def engage(self):
                pass

            def submit(self, target):
                calls.append(target)
                return Submission(target.command_id, True)

            def request_hold(self, reason):
                calls.append(reason)

        messages = []
        kinematics = SimpleNamespace(ik=lambda *args: (0.,)*7)
        with patch.object(tianji_jog, "TianjiCartesianExecutor", Executor), \
             patch.object(tianji_jog.time, "monotonic_ns", side_effect=count(now, 5_000_000)), \
             patch.object(tianji_jog.time, "sleep"):
            tianji_jog.run_jog(driver, kinematics, ControlProfile("p", "cartesian_impedance", {}),
                               side="left", translation_m=.005,
                               rotation_rad=math.radians(2), transition_s=.005,
                               terminal=_Keys("\r", "w", "i", "q"), emit=messages.append)
        targets = [item.tool_poses["left"] for item in calls if hasattr(item, "tool_poses")]
        self.assertEqual(len(targets), 3)
        self.assertEqual(targets[1].position_m, (.005, 0., 0.))
        self.assertEqual(targets[2].position_m, targets[1].position_m)
        self.assertAlmostEqual(targets[2].orientation_xyzw[0], math.sin(math.radians(1)))
        self.assertTrue(any("left W →" in message for message in messages))
        self.assertTrue(any("left I →" in message for message in messages))
        self.assertEqual(calls[-1], "keyboard jog exited")


class WujiHomeTests(unittest.TestCase):
    def test_trajectory_has_twenty_zero_endpoints(self):
        start = (.5,)*len(JOINT_NAMES)
        self.assertEqual(wuji_home.home_target(start, 0).position_rad, start)
        midpoint = wuji_home.home_target(start, 1.5).position_rad
        self.assertTrue(all(math.isclose(q, .25) for q in midpoint))
        self.assertEqual(wuji_home.home_target(start, 3).position_rad, (0.,)*20)
        self.assertEqual(wuji_home.home_target(start, 5).joint_names, JOINT_NAMES)

    def test_feedback_requires_fresh_valid_twenty_joint_frame(self):
        sample = Sample(SampleHeader(SampleRef("hand", "e", 1), 100, True),
                        JointState(JOINT_NAMES, (.01,)*20))
        self.assertAlmostEqual(wuji_home.feedback_error_rad(sample, after_ns=100), .01)
        self.assertIsNone(wuji_home.feedback_error_rad(sample, after_ns=101))
        self.assertIsNone(wuji_home.feedback_error_rad(
            Sample(SampleHeader(sample.header.ref, 100, False), sample.payload), after_ns=100))

    def test_rejected_home_command_disables(self):
        sample = Sample(SampleHeader(SampleRef("hand", "e", 1), time.monotonic_ns(), True),
                        JointState(JOINT_NAMES, (.2,)*20))
        calls = []

        class Hand:
            device_id = "wuji_left_hand"
            last_target = None
            timeout_ns = 500_000_000

            def configure(self, profile):
                self.profile = profile
                calls.append("configure")

            def engage(self):
                self.last_target = wuji_home.home_target((.2,)*20, 0)
                calls.append("engage")

            def get_latest(self):
                return sample

            def health(self, **kwargs):
                return SimpleNamespace(ready=True, detail=None)

            def submit(self, command):
                calls.append("submit")
                return Submission(command.command_id, False, "fault")

            def disable(self, reason):
                calls.append("disable")

        with self.assertRaisesRegex(RuntimeError, "fault"):
            wuji_home.run_home(Hand(), ControlProfile("p", "mit", {}), duration_s=.01,
                               terminal=_Keys(None), emit=lambda _: None)
        self.assertEqual(calls, ["configure", "engage", "submit", "disable"])

    def test_stale_diagnostics_disables_without_submission(self):
        calls = []

        class Hand:
            last_target = None
            timeout_ns = 500_000_000

            def configure(self, profile):
                calls.append("configure")

            def engage(self):
                self.last_target = wuji_home.home_target((.2,)*20, 0)
                calls.append("engage")

            def health(self, **kwargs):
                return SimpleNamespace(ready=False, detail="diagnostics timed out")

            def disable(self, reason):
                calls.append("disable")

        with self.assertRaisesRegex(RuntimeError, "diagnostics timed out"):
            wuji_home.run_home(Hand(), ControlProfile("p", "mit", {}), duration_s=.01,
                               terminal=_Keys(None), emit=lambda _: None)
        self.assertEqual(calls, ["configure", "engage", "disable"])


if __name__ == "__main__":
    unittest.main()
