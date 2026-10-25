"""Eye-in-hand: G_i @ X @ C_i = B, with fixed split and pixel refinement."""

from dataclasses import dataclass

import numpy as np

from .external import View, solve_external
from .geometry import checked_pose


@dataclass
class WristView:
    base_flange: np.ndarray
    pixels: np.ndarray
    group: int


def solve_wrist(views, camera, board=None, limits=None):
    # X @ C_i = inv(G_i) @ B has the same two-constant-transform objective
    # as the fixed-camera solver. The returned X is flange -> raw sensor.
    transformed = [View(np.linalg.inv(checked_pose(v.base_flange)), v.pixels, v.group) for v in views]
    return solve_external(transformed, camera, board, limits)
