"""Local action sources use simulation time and never access physical devices."""

from __future__ import annotations

import json
from pathlib import Path

from ._dependencies import np


class JointTrajectory:
    def __init__(self, path, model, runtime):
        document = json.loads(Path(path).read_text())
        self.test_data = bool(document.get("test_data", False))
        if document.get("joint_names") != list(model.names):
            raise ValueError("Trajectory joint_names must match the prepared robot's 54-joint order")
        points = document.get("waypoints", [])
        self.times = np.asarray([p["time_s"] for p in points], dtype=float)
        self.positions = np.asarray([p["positions_rad"] for p in points], dtype=float)
        if (len(self.times) < 2 or self.positions.shape != (len(points), 54)
                or not np.isfinite(self.positions).all() or not np.isfinite(self.times).all()
                or self.times[0] != 0 or np.any(np.diff(self.times) <= 0)):
            raise ValueError("Need at least two finite, ordered waypoints starting at time_s=0, with 54 positions each")
        if np.any(self.positions < model.lower-1e-9) or np.any(self.positions > model.upper+1e-9):
            raise ValueError("Trajectory exceeds joint limits")
        speeds = np.abs(np.diff(self.positions, axis=0))/np.diff(self.times)[:, None]
        limits = np.r_[np.full(14, runtime["joint_speed_rad_s"]), np.full(40, runtime["hand_speed_rad_s"])]
        if np.any(speeds > limits+1e-9):
            raise ValueError("Trajectory exceeds configured joint speed limits")

    def sample(self, simulation_seconds):
        time_s = float(np.clip(simulation_seconds, self.times[0], self.times[-1]))
        index = min(int(np.searchsorted(self.times, time_s, side="right"))-1, len(self.times)-2)
        fraction = (time_s-self.times[index])/(self.times[index+1]-self.times[index])
        return (1-fraction)*self.positions[index]+fraction*self.positions[index+1]
