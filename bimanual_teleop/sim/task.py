"""Physical placement diagnostics, kept out of policy observations."""

from __future__ import annotations

import itertools

from bimanual_teleop.sim._dependencies import np
from bimanual_teleop.sim._dependencies import Rotation

from .kinematics import pose_matrix


class TaskMonitor:
    def __init__(self, config, geometry):
        self.config, self.geometry = config, geometry
        self.reset()

    def reset(self):
        self.last_time = None; self.last_position = None
        self.fixture_since = None; self.bin_since = None
        self.fixture_completed = False

    def update(self, state):
        matrix = pose_matrix(state["rail"])
        position, rotation = matrix[:3, 3], matrix[:3, :3]
        stamp = state["time_ns"]/1e9
        speed = float("inf") if self.last_time is None or stamp <= self.last_time else float(np.linalg.norm(position-self.last_position)/(stamp-self.last_time))
        self.last_time, self.last_position = stamp, position.copy()
        contacts = state.get("hand_contact_force")
        released = contacts is not None and len(contacts) == 2 and all(np.isfinite(v) and v <= .05 for v in contacts)
        rail = self.config["rail"]
        local_bottoms = np.array([[-rail["hole_spacing_m"]/2, 0, -rail["size_m"][2]/2],
                                  [rail["hole_spacing_m"]/2, 0, -rail["size_m"][2]/2]])
        bottoms = local_bottoms @ rotation.T + position
        fixture = np.array(self.config["scene"]["fixture_origin"])
        pins = np.array(self.geometry["pin_centers_m"])+fixture
        radial = min(float(np.linalg.norm(bottoms[order, :2]-pins[:, :2], axis=1).max()) for order in ([0, 1], [1, 0]))
        angle = float(np.arccos(np.clip(abs(rotation[2, 2]), 0, 1)))
        bottom_z = float(bottoms[:, 2].mean())
        gap = abs(bottom_z-(fixture[2]+self.geometry["support_height_m"]))
        insertion = fixture[2]+self.geometry["pin_top_height_m"]-float(bottoms[:, 2].max())
        seated = radial <= .0015 and angle <= np.deg2rad(3) and gap <= .001 and insertion >= .005 and speed <= .01 and released
        self.fixture_since = (stamp if self.fixture_since is None else self.fixture_since) if seated else None
        if self.fixture_since is not None and stamp-self.fixture_since >= 1:
            self.fixture_completed = True
        corners = np.array(list(itertools.product(*[(-size/2, size/2) for size in rail["size_m"]]))) @ rotation.T + position
        bin_origin = np.asarray(self.config["scene"]["bin_centers"][1])
        size = np.asarray(self.config["scene"]["bin_inner_size"])
        inside = bool(np.all(np.abs(corners[:, :2]-bin_origin[:2]) <= size[:2]/2-.001)
                      and corners[:, 2].min() >= bin_origin[2]+self.config["scene"]["bin_wall"]-.001
                      and corners[:, 2].max() <= bin_origin[2]+size[2]+.001)
        finished = inside and released and speed <= .01 and self.fixture_completed
        self.bin_since = (stamp if self.bin_since is None else self.bin_since) if finished else None
        return {"pin_error_m": radial, "axis_error_rad": angle, "support_gap_m": gap, "insertion_depth_m": insertion,
                "hands_released": released, "fixture_seated": seated, "fixture_completed": self.fixture_completed,
                "task_completed": self.bin_since is not None and stamp-self.bin_since >= 1}
