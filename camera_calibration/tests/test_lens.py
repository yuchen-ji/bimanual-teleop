"""Native lens math is exercised without opening any RealSense device."""

import unittest
from unittest.mock import patch

import numpy as np
import pyrealsense2 as rs

from camera_calibration.board import (
    CircleBoard,
    camera_matrix,
    opencv_distortion,
    sensor_rays,
)
from camera_calibration.tests.test_external import lens


class LensTests(unittest.TestCase):
    def test_modified_brown_native_projection_roundtrip_without_context(self):
        metadata = lens("main-camera")
        metadata["rgb_intrinsics"]["model"] = "distortion.modified_brown_conrady"
        native = rs.intrinsics()
        native.width, native.height = metadata["resolution"]
        for name in ("fx", "fy", "ppx", "ppy"):
            setattr(native, name, metadata["rgb_intrinsics"][name])
        native.coeffs = metadata["rgb_intrinsics"]["coeffs"]
        native.model = rs.distortion.modified_brown_conrady
        pixels = np.array([[30.0, 40.0], [319.0, 241.0], [610.0, 440.0]])
        with patch(
            "pyrealsense2.context", side_effect=AssertionError("No device context")
        ):
            rays = sensor_rays(pixels, metadata)
            _, mapping_error = opencv_distortion(metadata)
        recovered = np.array(
            [rs.rs2_project_point_to_pixel(native, point.tolist()) for point in rays]
        )
        np.testing.assert_allclose(recovered, pixels, atol=1e-4)
        self.assertLess(mapping_error, 0.1)

    def test_nonfinite_and_mismatched_lens_metadata_fail(self):
        for change in (
            {"fx": float("nan")},
            {"fx": 0.0},
            {"width": 320},
            {"coeffs": [float("nan")] * 5},
        ):
            metadata = lens("camera")
            metadata["rgb_intrinsics"].update(change)
            with self.assertRaises(ValueError):
                camera_matrix(metadata)
        with self.assertRaises(ValueError):
            CircleBoard(same_row_spacing_m=float("nan")).points()
if __name__ == "__main__":
    unittest.main()
