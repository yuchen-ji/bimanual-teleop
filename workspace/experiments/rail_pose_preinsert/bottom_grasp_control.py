"""Continue a live supported scene from side grip to bottom-pad grip."""
import numpy as np
from locked_grasp import LockedGrasp


class BottomGraspControl(LockedGrasp):
    def __init__(self, source, state):
        self.release_pending = True
        super().__init__(source, state)
        self.phase = "release_review"
        self.opened = self.start.copy()
        self.opened[14:34] = source.runtime["open"]
        if source.task.get("bottom_recovery_open"):
            self.opened[14:18] = self.start[14:18]

    def request(self, stage, state):
        if self.release_pending:
            stage = "bottom_release_check"
            self.release_pending = False
        super().request(stage, state)

    def step(self, state, dt):
        if self.phase == "release_review":
            response = self.response()
            if response:
                self.phase, self.elapsed = "opening", 0.
                self.source.event("bottom_grasp_release_path_validated", validation=response)
            return self.source.command.copy()
        if self.phase == "opening":
            self.elapsed += dt
            u = min(self.elapsed/8., 1.)
            u = u*u*u*(10-15*u+6*u*u)
            desired = (1-u)*self.start+u*self.opened
            if self.elapsed >= 9.:
                self.rail_start = np.asarray(state["rail"]).copy()
                self.rail_xyz = self.rail_start[:3].copy()
                self.quaternion = self.rail_start[3:].copy()
                self.source.joint.CreateJointEnabledAttr(False)
                self.source.body.CreateKinematicEnabledAttr(True)
                self.source.ccd.CreateEnableCCDAttr(False)
                self.source.support_active = True
                self.source.grip = None
                self.start = self.opened.copy()
                self.phase, self.elapsed = "open_settle", 0.
                self.source.event("old_grasp_detached_orientation_locked")
            return desired
        if self.phase == "open_settle":
            self.elapsed += dt
            if self.elapsed >= 1.:
                self.phase, self.elapsed = "plan", 0.
                self.request("locked_grasp_plan", state)
            return self.opened.copy()
        return super().step(state, dt)
