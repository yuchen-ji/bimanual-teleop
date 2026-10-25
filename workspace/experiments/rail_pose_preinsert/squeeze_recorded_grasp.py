"""Prepare a small supported closure step from measured PhysX link poses."""
import argparse
import json
import sys
from pathlib import Path

import fcl
import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[2])]
from bottom_grasp import BottomGraspFit
from common import finger_forces, write_json
from four_finger_geometry import FourFingerGeometry
from grasp_fit import convex
from locked_grasp import FINGERS, distal_forces

from bimanual_teleop.sim.config import ROOT
from bimanual_teleop.sim.kinematics import RobotModel


def prepare(state, identifier, target_force=2., collision_model=None):
    assets = ROOT/'assets/generated'
    model = RobotModel(assets/'robot.urdf')
    fit = BottomGraspFit(model, assets/'robot.urdf', assets/'manifest.json', state)
    if collision_model is not None:
        cooked = json.loads(Path(collision_model).read_text())
        for name, record in cooked.items():
            if name not in fit.objects:
                continue
            vertices = trimesh.transform_points(record['convexes'][0]['vertices'],
                                                np.asarray(record['mesh_to_link']).T)
            mesh = trimesh.Trimesh(vertices=vertices).convex_hull
            fit.objects[name] = fcl.CollisionObject(convex(mesh))
            for finger in FINGERS[1:]:
                if name == 'l_'+finger+'_tip_sensor_frame':
                    fit.pad_vertices[finger] = vertices
    geometry = FourFingerGeometry(model, assets/'robot.urdf', assets/'manifest.json').check(state)
    q = np.asarray(state['q'])
    command = np.asarray(state['q_target']).copy()
    xyz = np.asarray(state['rail'][:3]).copy()
    force = distal_forces(state)
    force['thumb'] = finger_forces(state)['thumb']
    ds = fit.distances(q, xyz)
    self_gap = fit.self_distances().min()
    wrist = fit.names.index('l_wrist')
    palm_force = state.get('rail_contacts', {}).get('l_wrist', 0.)
    report = {'geometry': geometry, 'force_n': force, 'palm_force_n': palm_force,
              'cooked_collision_model': str(collision_model) if collision_model else None,
              'thumb_force_includes_all_segments': True,
              'envelope_gaps_m': dict(zip(fit.names, ds.tolist())), 'self_envelope_gap_m': float(self_gap)}
    if geometry['interfinger_intersections'] or self_gap < -.00005 or ds.min() < -.00005:
        report['stop'] = 'Measured collision envelope exceeds 0.05 mm contact-solver tolerance'
        return None, report
    # Move the rail upward toward the palm only; its orientation is unchanged.
    if palm_force > 3.:
        xyz -= .00004*fit.rail_pose[:3, 2]
    elif palm_force < 1.5:
        shift = min(.00004, max(.000004, .5*ds[wrist]))
        xyz += shift*fit.rail_pose[:3, 2]
    for i, finger in enumerate(FINGERS):
        desired_force = .7 if finger == 'thumb' else target_force
        if force[finger] >= desired_force:
            continue
        indices = np.arange(14, 18) if i == 0 else np.array([14+4*i, 16+4*i, 17+4*i])
        def gap(values, i=i, finger=finger):
            if i == 0:
                return fit.distances(values, xyz)[fit.tips[0]]
            return fit.bottom_metrics(values, xyz)[finger]['gap_m']
        distance = gap(q)
        gradient = []
        for index in indices:
            moved = q.copy(); moved[index] += .00005
            gradient.append((gap(moved)-distance)/.00005)
        gradient = np.asarray(gradient)
        norm2 = float(gradient@gradient)
        if norm2 < 1e-12:
            continue
        advance = min(.00007, max(.00005, .6*distance))
        delta = -advance*gradient/norm2
        delta *= min(1., .003/max(np.linalg.norm(delta), 1e-12))
        root_name = 'l_'+finger+'_proximal_abd'
        if i > 0 and state.get('rail_contacts', {}).get(root_name, 0.) > 3.:
            # Relieve knuckle pressure while closing the distal pad. A fixed
            # MCP angle would let the knuckle block the fingertip's approach.
            indices = np.arange(14+4*i, 18+4*i)
            base = fit.distances(q, xyz)[fit.names.index(root_name)]
            root_gradient, bottom_gradient = [], []
            for index in indices:
                moved = q.copy(); moved[index] += .00005
                root_gradient.append((fit.distances(moved, xyz)[fit.names.index(root_name)]-base)/.00005)
                bottom_gradient.append((gap(moved)-distance)/.00005)
            matrix = np.array([root_gradient, bottom_gradient, [.02, 0., .02, .02]])
            delta = matrix.T @ np.linalg.solve(matrix@matrix.T+1e-10*np.eye(3), [.00005, -.00004, 0.])
            delta *= min(1., .006/max(np.linalg.norm(delta), 1e-12))
        command[indices] += delta
    command = np.clip(command, model.lower, model.upper)
    # The final acceptance is an unsupported hold, never the force threshold alone.
    report['ready_for_hold'] = palm_force >= .8 and all(force[f] >= .7 for f in FINGERS) and sum(force[f] for f in FINGERS[1:]) >= 7.8
    request = {'id': identifier, 'q': command.tolist(), 'rail': [*xyz.tolist(), *state['rail'][3:]],
               'supported': True, 'duration_s': .3, 'settle_s': .7, 'force_limit_n': 4.}
    return request, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('state', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--id', required=True)
    parser.add_argument('--collision-model', type=Path)
    args = parser.parse_args()
    request, report = prepare(json.loads(args.state.read_text()), args.id, collision_model=args.collision_model)
    write_json(args.output/(args.id+'_geometry.json'), report)
    if request:
        write_json(args.output/'segment.json', request)
    print(json.dumps({k: v for k, v in report.items() if k not in ('geometry', 'envelope_gaps_m')}, indent=2))


if __name__ == '__main__':
    main()
