"""Measured distal-surface checks and conservative refinement for a grasp latch."""
import numpy as np
from four_finger_geometry import FourFingerGeometry
from grasp_fit import FINGERS, GraspFit
from locked_grasp import distal_forces

from bimanual_teleop.sim.kinematics import pose_matrix


def handle(active, request):
    model = active.geometry.robot
    state = request["state"]
    bottom = getattr(active, "task", {}).get("grasp_contact_surface") == "bottom"
    if bottom:
        from bottom_grasp import BottomGraspFit
    fit = (BottomGraspFit if bottom else GraspFit)(model, active.config["urdf"], active.config["manifest"], state)
    if request["stage"] == "locked_grasp_plan":
        return {"ok": True, "plan": fit.plan()}
    checker = FourFingerGeometry(model, active.config["urdf"], active.config["manifest"])
    geometry = checker.check(state)
    command = np.asarray(request["command"]).copy()
    q = np.asarray(state["q"])
    rail = pose_matrix(state["rail"])

    def distance(finger, joints):
        if bottom and finger != "thumb":
            return fit.bottom_metrics(joints, rail[:3, 3])[finger]["gap_m"]
        transforms = fit.transforms(joints)
        manager = checker.distal[finger]
        for suffix in ("distal", "tip_sensor_frame"):
            name = "l_"+finger+"_"+suffix
            manager.set_transform(name, transforms[name])
        return float(manager.min_distance_single(checker.rail, rail))

    for i, finger in enumerate(FINGERS):
        gap = distance(finger, q) if bottom and finger != "thumb" else geometry["distal_gap_m"][finger]
        if .00007 <= gap <= .00018:
            continue
        indices = np.arange(14, 18) if i == 0 else np.array([14+4*i, 16+4*i, 17+4*i])
        gradient = []
        for j in indices:
            moved = q.copy(); moved[j] += .00005
            gradient.append((distance(finger, moved)-gap)/.00005)
        gradient = np.asarray(gradient)
        scale = np.linalg.norm(gradient)
        if scale > 1e-8:
            # Close only halfway through the measured excess gap on each
            # iteration, leaving a positive margin for tracking error.
            step = np.clip(.5*(gap-.00012)/scale, -.002, .004)
            command[indices] -= step*gradient/scale
    command = np.clip(command, model.lower, model.upper)
    return {"ok": True, "geometry": geometry, "command": command.tolist(), "distal_force_n": distal_forces(state)}
