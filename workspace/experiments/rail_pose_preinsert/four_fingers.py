"""Incremental supported closure with settling, geometry review and retreat."""
from __future__ import annotations

import numpy as np
from common import finger_forces

FINGERS = ("index_finger", "middle_finger", "ring_finger", "pinky")


class FourFingerClosure:
    def __init__(self, command, grasp, duration, initial_state=None):
        self.start = np.asarray(command, dtype=float).copy()
        self.target = self.start.copy()
        self.indices = [np.array([18 + 4*i, 20 + 4*i, 21 + 4*i]) for i in range(4)]
        for indices in self.indices:
            self.target[indices] = np.asarray(grasp)[indices - 14]
        self.command = self.start.copy()
        self.safe = self.start.copy()
        self.previous_safe = self.start.copy()
        self.origin = self.start.copy()
        self.goal = self.start.copy()
        self.phase = "check"  # Validate the initial pose before any motion.
        self.elapsed = 0.
        self.move_s = max(.35, duration / 10)
        self.step_rad = np.full(4, np.radians(2.5))
        self.blocked = set()
        self.touched = set()
        self.forces = dict.fromkeys(FINGERS, 0.)
        self.gaps = dict.fromkeys(FINGERS, float("inf"))
        self.stable = 0.
        self.done = False
        self.cycles = 0
        self.events = []
        self.initial_eef = np.array(initial_state["eef"][0]) if initial_state and "eef" in initial_state else None
        self.last_review = None
        self.failure = None

    def retreat(self, affected, reason):
        self.origin = self.command.copy()
        self.goal = self.command.copy()
        for i in affected:
            indices = self.indices[i]
            fallback = self.previous_safe if np.allclose(self.command[indices], self.safe[indices], atol=1e-10) else self.safe
            self.goal[indices] = fallback[indices]
            if self.step_rad[i] <= np.radians(.1001):
                self.blocked.add(i)
            self.step_rad[i] = np.radians(.5 if self.step_rad[i] > np.radians(.5001) else .1)
        self.phase, self.elapsed = "retreat", 0.
        self.events.append({"kind": "four_finger_retreat", "fingers": [FINGERS[i] for i in affected], "reason": reason})

    def review(self, report):
        if self.phase != "check":
            return  # A force-triggered retreat invalidates an older geometry reply.
        self.last_review = report
        if report["intersecting_links"]:
            if self.cycles == 0:
                self.failure = "Initial hand/rail mesh intersection; closure was not started"
                return
            affected = [i for i, f in enumerate(FINGERS)
                        if any(name.startswith("l_"+f+"_") for name in report["intersecting_links"])]
            if not affected:
                affected = list(range(4))
                self.blocked.update(affected)
            self.retreat(affected, "Visual mesh intersection: " + ", ".join(report["intersecting_links"]))
            return
        if not np.allclose(self.safe, self.command, atol=1e-10):
            self.previous_safe = self.safe.copy()
            self.safe = self.command.copy()
        self.gaps = report["finger_gap_m"]
        self.cycles += 1
        self.events.append({"kind": "four_finger_checkpoint", "cycle": self.cycles,
                            "force_n": dict(self.forces), "gap_m": dict(self.gaps),
                            "hand_command_rad": self.command[14:34].tolist()})
        self.origin = self.command.copy()
        self.goal = self.command.copy()
        for i, (finger, indices) in enumerate(zip(FINGERS, self.indices)):
            if i in self.blocked or self.forces[finger] >= .2:
                continue
            if self.gaps[finger] < .002 or i in self.touched:
                self.step_rad[i] = min(self.step_rad[i], np.radians(.5))
            self.goal[indices] += np.clip(self.target[indices]-self.command[indices], -self.step_rad[i], self.step_rad[i])
        if np.allclose(self.goal, self.command, atol=1e-10) or self.cycles >= 240:
            if all(self.forces[f] >= .2 for f in FINGERS) and self.stable < .5:
                self.phase, self.elapsed = "settle", 0.
                return
            self.done = True
            self.phase = "done"
        else:
            self.phase, self.elapsed = "move", 0.

    def step(self, state, dt):
        forces = finger_forces(state)
        self.forces = {f: forces[f] for f in FINGERS}
        self.stable = self.stable + dt if all(forces[f] >= .2 for f in FINGERS) else 0.
        if self.done:
            return self.command.copy()
        if self.phase != "retreat":
            overloaded = [i for i, f in enumerate(FINGERS) if forces[f] > 3.]
            displaced = (self.initial_eef is not None and "eef" in state
                         and np.linalg.norm(np.asarray(state["eef"][0])[:3]-self.initial_eef[:3]) > .001)
            if overloaded or displaced:
                self.retreat(list(range(4)) if displaced else overloaded,
                             "Hand displaced over 1 mm" if displaced else "Contact force exceeded 3 N")
        self.elapsed += dt
        if self.phase in ("move", "retreat"):
            u = min(self.elapsed/self.move_s, 1.)
            blend = 10*u**3-15*u**4+6*u**5
            candidate = self.origin + blend*(self.goal-self.origin)
            for i, (finger, indices) in enumerate(zip(FINGERS, self.indices)):
                if self.phase == "move" and forces[finger] >= .3:
                    self.touched.add(i)
                    self.goal[indices] = self.command[indices]
                    self.origin[indices] = self.command[indices]
                    candidate[indices] = self.command[indices]
            self.command = candidate
            if u >= 1:
                self.phase, self.elapsed = "settle", 0.
        elif self.phase == "settle" and self.elapsed >= .6:
            self.phase, self.elapsed = "check", 0.
        return self.command.copy()
