"""Measured micro-closure toward force contact, with original mesh checks."""
import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[2])]
from bottom_grasp import BottomGraspFit
from four_finger_geometry import FourFingerGeometry
from host_rpc import RequestServer, WorkerLease, process_identity
from locked_grasp import FINGERS, distal_forces

from bimanual_teleop.sim.config import ROOT
from bimanual_teleop.sim.kinematics import RobotModel


def handle(active, request):
    state = request['state']
    fit = BottomGraspFit(active.model, active.urdf, active.manifest, state)
    checker = FourFingerGeometry(active.model, active.urdf, active.manifest)
    geometry = checker.check(state)
    q = np.asarray(state['q'])
    command = np.asarray(request['command']).copy()
    xyz = fit.rail_pose[:3, 3]
    force = distal_forces(state)
    bottom = fit.bottom_metrics(q, xyz)
    geometry['bottom_plane_signed_gap_m'] = {f: v['gap_m'] for f, v in bottom.items()}
    # Collision is reported before issuing any further closure. No relaxed
    # penetration threshold replaces the original full-mesh acceptance check.
    if geometry['intersecting_links'] or geometry['interfinger_intersections']:
        return {'ok': True, 'geometry': geometry, 'command': command.tolist(), 'stop': 'Mesh intersection during tightening', 'force_n': force}

    def gap(finger, moved):
        if finger != 'thumb':
            return fit.bottom_metrics(moved, xyz)[finger]['gap_m']
        transforms = fit.transforms(moved)
        manager = checker.distal['thumb']
        for suffix in ('distal', 'tip_sensor_frame'):
            name = 'l_thumb_'+suffix
            manager.set_transform(name, transforms[name])
        return float(manager.min_distance_single(checker.rail, fit.rail_pose))

    # First remove the visible clearance in <=30 micrometre steps. At the
    # surface, use <=5 micrometre increments and stop on the first intersection.
    targets = dict.fromkeys(FINGERS, 2.)
    targets['thumb'] = 1.5
    for i, finger in enumerate(FINGERS):
        if force[finger] >= targets[finger]:
            continue
        distance = gap(finger, q)
        indices = np.arange(14, 18) if i == 0 else np.array([14+4*i, 16+4*i, 17+4*i])
        gradient = []
        for j in indices:
            moved = q.copy(); moved[j] += .00002
            gradient.append((gap(finger, moved)-distance)/.00002)
        gradient = np.asarray(gradient)
        norm2 = float(gradient@gradient)
        if norm2 > 1e-12:
            advance = min(.00003, max(.000005, .5*distance))
            delta = -advance*gradient/norm2
            delta *= min(1., .001/max(np.linalg.norm(delta), 1e-12))
            command[indices] += delta
    command = np.clip(command, active.model.lower, active.model.upper)
    return {'ok': True, 'geometry': geometry, 'command': command.tolist(), 'force_n': force,
            'force_targets_n': targets, 'ready': all(force[f] >= targets[f] for f in FINGERS)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--scene-pid', type=int, required=True)
    args = parser.parse_args()
    p = ROOT/'assets/generated'
    active = SimpleNamespace(model=RobotModel(p/'robot.urdf'), urdf=p/'robot.urdf', manifest=p/'manifest.json')
    identity = process_identity(args.scene_pid)
    if identity is None:
        parser.error('Isaac scene process is not alive')
    with WorkerLease(args.output, args.scene_pid) as lease:
        server = RequestServer(args.output, lambda request: handle(active, request), lease)
        while process_identity(args.scene_pid) == identity:
            server.poll()
            result = args.output/'tighten_result.json'
            if result.exists() and json.loads(result.read_text()).get('complete'):
                break
            time.sleep(.05)


if __name__ == '__main__':
    main()
