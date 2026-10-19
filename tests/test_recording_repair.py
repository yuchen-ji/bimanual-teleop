"""Repair actual encoded video gaps without inventing control labels."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import zarr

from bimanual_teleop.recording.convert import convert_recordings
from tests.support.recording import make_episode, START_NS


class RecordingRepairTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def convert(self, **config):
        path = self.root / "out.zarr"
        report = convert_recordings(self.root / "raw", path, action_space="eef",
                                    conversion_config=config)
        return zarr.open_group(str(path), mode="r"), report

    def test_one_and_two_missing_frames_reuse_past_pixels(self):
        times = np.arange(9) * 1000 / 30
        make_episode(self.root / "raw", main=times,
                     other={1: times[[0, 1, 3, 4, 7, 8]], 2: times[[0, 1, 3, 4, 7, 8]]})
        data, report = self.convert()
        np.testing.assert_allclose(np.diff(data['data/timestamp'][:]), 1/30, atol=1e-9)
        self.assertEqual(report['output_segments'], 1)
        np.testing.assert_array_equal(data['meta/camera_reused'][:, 1], [0, 0, 1, 0, 0, 1, 1, 0, 0])
        np.testing.assert_array_equal(data['data/camera_1'][2], data['data/camera_1'][1])
        self.assertEqual(data['meta/camera_source_row'][2, 1], 1)
        self.assertLessEqual(data['meta/command_age_ns'][:].max(), 50_000_000)

    def test_three_frame_hole_is_entirely_rejected_and_limit_is_configurable(self):
        times = np.arange(9) * 1000 / 30
        make_episode(self.root / 'raw', main=times, other={1: times[[0, 1, 5, 6, 7, 8]]})
        data, report = self.convert()
        self.assertEqual(report['output_segments'], 2)
        np.testing.assert_array_equal(data['meta/camera_reused'][:, 1], 0)
        self.assertEqual(report['output_frames'], 6)
        path = self.root / 'expanded.zarr'
        report = convert_recordings(self.root/'raw', path, action_space='eef',
                                    conversion_config={'max_missing_camera_frames': 3})
        self.assertEqual(report['output_segments'], 1)
        self.assertEqual(report['output_frames'], 9)

    def test_main_camera_hole_is_gridded_and_edges_are_not_extended(self):
        times = np.arange(8) * 1000 / 30
        make_episode(self.root/'raw', main=times[[0, 1, 3, 4, 5, 6, 7]],
                     other={1: times[1:], 2: times})
        data, report = self.convert()
        self.assertEqual(report['output_frames'], 7)
        self.assertEqual(report['output_segments'], 1)
        self.assertTrue(data['meta/camera_reused'][1, 0])
        self.assertAlmostEqual(data['data/timestamp'][0], 1/30)

    def pause_episode(self):
        times = np.arange(9) * 1000 / 30
        _, raw = make_episode(self.root/'raw', main=times)
        for camera in range(3):
            raw[f'cameras/camera_{camera}/rgb/source_time_ms'][4:] += 2000
        return raw

    def test_unreviewed_pause_isolated_and_approved_seam_compressed(self):
        self.pause_episode()
        data, report = self.convert()
        self.assertEqual(report['output_segments'], 2)
        self.assertFalse(report['episodes'][0]['pauses'][0]['merged'])
        before = data['meta/recording_time_ns'][:]
        path = self.root/'merged.zarr'
        report = convert_recordings(self.root/'raw', path, action_space='eef', conversion_config={
            'pause_reviews': {'episode_000000': {'4': {'merge': True, 'reason': 'Synthetic continuous seam'}}}})
        merged = zarr.open_group(str(path), mode='r')
        self.assertEqual(report['output_segments'], 1)
        np.testing.assert_allclose(np.diff(merged['data/timestamp'][:]), 1/30, atol=1e-9)
        np.testing.assert_array_equal(merged['meta/recording_time_ns'][:], before)
        self.assertGreater(np.diff(merged['meta/source_time_ns'][:]).max(), 2_000_000_000)
        # Seam labels are kept from each real side, not blended across pause.
        np.testing.assert_array_equal(merged['data/action'][:], data['data/action'][:])

    def test_pause_does_not_make_an_old_command_fresh(self):
        raw = self.pause_episode()
        # Last pre-pause command cannot be borrowed by the next recording block.
        for side in ('left', 'right'):
            for kind in ('arm_commands', 'hand_commands'):
                group = raw[f'{kind}/{side}']
                retained = group['time_ns'][:] < START_NS + 130_000_000
                for key in list(group.array_keys()):
                    array = group[key]
                    values = array[:][retained]
                    array.resize(values.shape)
                    array[:] = values
        data, report = self.convert(pause_reviews={'episode_000000': {4: {'merge': True, 'reason': 'Synthetic'}}})
        self.assertEqual(report['output_segments'], 1)
        self.assertFalse(report['episodes'][0]['pauses'][0]['merged'])
        self.assertTrue((data['meta/recording_block'][:] == 0).all())

    def test_state_invalidity_and_gap_are_not_skipped_or_extrapolated(self):
        times = np.arange(7) * 1000 / 30
        _, raw = make_episode(self.root/'raw', main=times, state=(0, 20, 40, 160, 180, 200))
        raw['arms/left/eef_pose'][1, 3:] = 0
        _, report = self.convert()
        reasons = report['episodes'][0]['invalid_reasons']
        self.assertGreater(reasons['left_robot_eef_pose_invalid_or_gap'], 0)
        self.assertGreater(reasons['right_robot_joint_invalid_or_gap'], 0)

    def test_short_segments_are_left_out_and_reported(self):
        times = np.arange(9) * 1000 / 30
        make_episode(self.root / 'raw', main=times, other={1: times[[0, 1, 5, 6, 7, 8]]})
        data, report = self.convert(min_segment_frames=3)
        self.assertEqual(report['output_frames'], 4)
        self.assertEqual(report['episodes'][0]['short_segments'], {'segments': 1, 'frames': 2})
        np.testing.assert_array_equal(data['meta/segment_ends'][:], [4])
        np.testing.assert_array_equal(data['meta/recording_time_ns'][:] - START_NS,
                                      np.rint(times[5:] * 1e6).astype(np.int64))
        with self.assertRaisesRegex(ValueError, 'min_segment_frames'):
            self.convert(min_segment_frames=-1)

    def test_strict_mode_from_config_keeps_strict_output_and_filters_short_segments(self):
        _, raw = make_episode(self.root / 'raw', main=(10, 40, 70, 100, 130, 160),
                              other={1: (10, 40, 100, 130, 160)})
        raw['arms/left/wrench'][13, 0] = np.nan
        data, report = self.convert(mode='strict', min_segment_frames=2)
        np.testing.assert_allclose(data['data/timestamp'][:], [.01, .04])
        self.assertEqual(report['episodes'][0]['short_segments'], {'segments': 2, 'frames': 2})
        self.assertEqual(report['conversion_config']['mode'], 'strict')
        self.assertNotIn('recording_time_ns', data['meta'])
        self.assertIn('camera_0 real frame times', data.attrs['timestamp'])

    def test_default_is_the_project_repair_config_and_pauses_split_without_review(self):
        from bimanual_teleop.common.config import load_yaml_config
        from bimanual_teleop.recording.policy import DEFAULT_CONFIG

        self.pause_episode()
        report = convert_recordings(self.root / 'raw', self.root / 'default.zarr', action_space='eef')
        expected = load_yaml_config(DEFAULT_CONFIG)
        self.assertEqual({key: report['conversion_config'][key] for key in expected}, expected)
        self.assertEqual(report['conversion_config']['mode'], 'repair')
        self.assertFalse(report['episodes'][0]['pauses'][0]['merged'])

    def test_explicit_10hz_config_exports_from_30hz_camera_recordings(self):
        # High-rate measured state and command streams remain available while
        # all output observations and absolute action labels share a 10 Hz grid.
        times = np.arange(61) * 1000 / 30
        make_episode(self.root / 'raw', main=times, state=np.arange(0, 2021, 10),
                     commands=np.arange(0, 2021, 20))
        path = self.root / '10hz.zarr'
        report = convert_recordings(self.root / 'raw', path, action_space='eef',
                                    conversion_config={'target_fps': 10})
        data = zarr.open_group(str(path), mode='r')
        self.assertEqual(data.attrs['target_fps'], 10)
        self.assertEqual(report['output_segments'], 1)
        np.testing.assert_allclose(np.diff(data['data/timestamp'][:]), .1, atol=1e-9)
        np.testing.assert_array_equal(data['meta/camera_source_row'][:, 0], np.arange(0, 61, 3))
        self.assertLessEqual(data['meta/command_age_ns'][:].max(), 50_000_000)

    def test_report_says_what_was_dropped_and_how_frames_were_repaired(self):
        from bimanual_teleop.cli.convert_recording import summary_lines

        times = np.arange(9) * 1000 / 30
        make_episode(self.root / 'raw', main=times, other={1: times[[0, 1, 3, 4, 5, 6, 7, 8]]},
                     state=(0, 20, 40, 120, 140, 160, 180, 200, 220, 240, 260, 280))
        _, report = self.convert()
        item = report['episodes'][0]
        self.assertEqual(item['state_repairs'], {'frames': 2, 'beyond_ms': 50.0, 'max_gap_ms': 80.0})
        self.assertEqual(item['camera_repairs'][1]['reused_frames'], 1)
        text = '\n'.join(summary_lines(report))
        self.assertIn('缺帧沿用前一张图：camera_1 1 帧（图像最旧 33 ms）', text)
        self.assertIn('放宽插值 2 帧（前后样本相隔超过 50 ms，最大 80 ms）', text)
        self.assertIn('修复：缺帧沿用前一张图 camera_1 1 帧；放宽插值 2 帧；暂停在此切开 0 处', text)

    def test_bad_config_and_clock_reset_do_not_publish(self):
        raw = self.pause_episode()
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            self.convert(reszie_enabled=True)
        raw['cameras/camera_0/rgb/source_time_ms'][5] = -1
        with self.assertRaisesRegex(ValueError, 'reset'):
            self.convert()
        self.assertFalse((self.root/'out.zarr').exists())
        self.assertFalse(list(self.root.glob('.*.converting-*')))


if __name__ == '__main__':
    unittest.main()
