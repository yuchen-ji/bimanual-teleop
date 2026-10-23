"""Strict and repair conversion outputs stay identical to the recorded baseline.

Regenerate only for an intended output change:
``python tests/test_recording_golden.py --regenerate``.
"""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from tests.support.recording import make_episode


FIXTURES = Path(__file__).resolve().parent / "fixtures"
ARRAYS = FIXTURES / "recording_conversion_golden.npz"
ATTRIBUTES = FIXTURES / "recording_conversion_golden.json"
REPORT_KEYS = ("source_episode", "status", "reference_frames", "valid_frames", "segments",
               "main_camera_gaps", "invalid_reasons", "pauses", "camera_repairs")


def _strict_input(root):
    _, raw = make_episode(root, main=(10, 40, 70, 100, 130, 160),
                          other={1: (10, 40, 100, 130, 160)}, depth=True)
    raw["arms/left/wrench"][13, 0] = np.nan
    make_episode(root, name="episode_000001", main=(10, 40, 120, 150), depth=True)
    failed = root / "episode_000002"
    failed.mkdir()
    (failed / "episode.json").write_text(json.dumps({"status": "failed"}))


def _repair_reuse_input(root):
    times = np.arange(9) * 1000 / 30
    make_episode(root, main=times,
                 other={1: times[[0, 1, 3, 4, 7, 8]], 2: times[[0, 1, 3, 4, 7, 8]]})
    make_episode(root, name="episode_000001", main=times[[0, 1, 3, 4, 5, 6, 7]],
                 other={1: times[1:8], 2: times[:8]})
    make_episode(root, name="episode_000002", main=times, other={1: times[[0, 1, 5, 6, 7, 8]]})


def _repair_pause_input(root):
    times = np.arange(9) * 1000 / 30
    sources = times.copy()
    sources[4:] += 2000
    make_episode(root, main=times, camera_sources={camera: sources for camera in range(3)})
    _, raw = make_episode(root, name="episode_000001", main=np.arange(7) * 1000 / 30,
                          state=(0, 20, 40, 160, 180, 200))
    raw["arms/left/eef_pose"][1, 3:] = 0


CASES = {
    "strict_eef_depth": (_strict_input, dict(action_space="eef", include_depth=True,
                                             conversion_config={"mode": "strict"})),
    "strict_joint": (_strict_input, dict(action_space="joint", conversion_config={"mode": "strict"})),
    "repair_reuse": (_repair_reuse_input, dict(action_space="eef", conversion_config={})),
    "repair_pause": (_repair_pause_input, dict(action_space="joint", conversion_config={
        "pause_reviews": {"episode_000000": {"4": {"merge": True, "reason": "Synthetic seam"}}}})),
}


def _json(value):
    return json.loads(json.dumps(value))


def convert_case(name, root):
    import zarr
    from bimanual_teleop.recording.convert import convert_recordings

    build, options = CASES[name]
    source = root / name / "raw"
    build(source)
    output = root / name / "out.zarr"
    report = convert_recordings(source, output, **options)
    dataset = zarr.open_group(str(output), mode="r")
    arrays = {}
    for group in ("data", "meta"):
        for key in sorted(dataset[group].array_keys()):
            arrays[f"{name}/{group}/{key}"] = np.asarray(dataset[group][key][:])
    attributes = {
        "root": _json(dict(dataset.attrs)),
        "segments": _json(dataset["meta"].attrs["segments"]),
        "report": {key: _json(report[key]) for key in (
            "action_space", "include_depth", "conversion_config",
            "output_episodes", "output_segments", "output_frames")},
        "episodes": [{key: _json(item[key]) for key in REPORT_KEYS if key in item}
                     for item in report["episodes"]],
    }
    return arrays, attributes


def regenerate():
    arrays, attributes = {}, {}
    with tempfile.TemporaryDirectory() as directory:
        for name in CASES:
            case_arrays, attributes[name] = convert_case(name, Path(directory))
            arrays.update(case_arrays)
    FIXTURES.mkdir(exist_ok=True)
    np.savez_compressed(ARRAYS, **arrays)
    ATTRIBUTES.write_text(json.dumps(attributes, indent=1, sort_keys=True) + "\n", encoding="utf-8")


@unittest.skipUnless(importlib.util.find_spec("av") and importlib.util.find_spec("zarr"),
                     "recording dependencies are not installed")
class RecordingGoldenTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.expected_arrays = dict(np.load(ARRAYS))
        self.expected_attributes = json.loads(ATTRIBUTES.read_text(encoding="utf-8"))

    def test_outputs_match_baseline(self):
        for name in CASES:
            with self.subTest(case=name):
                arrays, attributes = convert_case(name, Path(self.temporary.name))
                expected = {key: value for key, value in self.expected_arrays.items()
                            if key.startswith(f"{name}/")}
                self.assertEqual(sorted(arrays), sorted(expected))
                for key, value in arrays.items():
                    reference = expected[key]
                    self.assertEqual((value.dtype, value.shape), (reference.dtype, reference.shape), key)
                    if value.dtype.kind == "f":
                        np.testing.assert_allclose(value, reference, rtol=1e-6, atol=1e-6, err_msg=key)
                    else:
                        np.testing.assert_array_equal(value, reference, err_msg=key)
                self.assertEqual(attributes, self.expected_attributes[name])


if __name__ == "__main__":
    if sys.argv[1:] == ["--regenerate"]:
        regenerate()
    else:
        unittest.main()
