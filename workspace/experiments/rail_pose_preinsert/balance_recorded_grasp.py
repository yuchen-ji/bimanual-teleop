"""Prepare the bounded preload/balance steps used by the recorded grasp trial."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[2])]
from bottom_grasp import BottomGraspFit
from common import write_json

from bimanual_teleop.sim.config import ROOT
from bimanual_teleop.sim.kinematics import RobotModel


def prepare(state, identifier, operation):
    assets = ROOT/'assets/generated'
    fit = BottomGraspFit(RobotModel(assets/'robot.urdf'), assets/'robot.urdf', assets/'manifest.json', state)
    q = np.asarray(state['q']).copy()
    xyz = state['rail'][:3]
    target = np.asarray(state['q_target']).copy()
    reference = np.asarray(json.loads((HERE/'hand_profiles/recorded_left_tight_grasp.json').read_text())['measured_position_rad'])
    fit.distances(q, xyz)
    self_gap = fit.self_distances().min()
    if operation == 'preload':
        # Separate adjacent pads before increasing their drive preload.
        best = (self_gap, 0., 0.)
        for index_abd in [-.02, -.01, 0., .01, .02]:
            for middle_abd in [-.01, 0., .01]:
                trial = q.copy(); trial[19] += index_abd; trial[23] += middle_abd
                fit.distances(trial, xyz)
                gap = fit.self_distances().min()
                if gap > best[0]:
                    best = (gap, index_abd, middle_abd)
        if self_gap < .0002:
            q[[19, 23]] += best[1:]
            target[[19, 23]] += best[1:]
        for finger, start in zip(('index_finger', 'middle_finger', 'ring_finger', 'pinky'), (18, 22, 26, 30)):
            indices = np.array([start, start+2, start+3])
            gap = fit.bottom_metrics(q, xyz)[finger]['gap_m']
            gradient = []
            for joint in indices:
                trial = q.copy(); trial[joint] += .00005
                gradient.append((fit.bottom_metrics(trial, xyz)[finger]['gap_m']-gap)/.00005)
            gradient = np.asarray(gradient)
            if gradient@gradient < 1e-12:
                raise ValueError('Degenerate distal-pad Jacobian')
            delta = -.0005*gradient/(gradient@gradient)
            delta *= min(1., .015/np.linalg.norm(delta))
            target[indices] += delta
        thumb_force = sum(force for name, force in state['rail_contacts'].items() if name.startswith('l_thumb_'))
        if thumb_force < 2.:
            target[14:18] += .05*(reference[:4]-q[14:18])
        duration, settle = 1., .7
    elif operation == 'balance':
        # Open the MCP slightly while compensating at PIP/DIP: relieve root
        # pressure without withdrawing the distal pads from the underside.
        for finger, start in [('index_finger', 18), ('pinky', 30)]:
            root = fit.names.index('l_'+finger+'_proximal_abd')
            indices = np.array([start, start+2, start+3])
            distance = fit.distances(q, xyz)[root]
            bottom = fit.bottom_metrics(q, xyz)[finger]['gap_m']
            root_gradient, bottom_gradient = [], []
            for joint in indices:
                trial = q.copy(); trial[joint] += .00005
                root_gradient.append((fit.distances(trial, xyz)[root]-distance)/.00005)
                bottom_gradient.append((fit.bottom_metrics(trial, xyz)[finger]['gap_m']-bottom)/.00005)
            matrix = np.array([root_gradient, bottom_gradient, [.02, .02, .02]])
            delta = np.linalg.lstsq(matrix, [.0006, 0., 0.], rcond=1e-5)[0]
            delta *= min(1., .04/max(np.linalg.norm(delta), 1e-12))
            target[indices] += delta
        target[14:18] += .2*(reference[:4]-q[14:18])
        duration, settle = 1.5, .8
    else:
        raise ValueError('Unknown preload operation')
    target = np.clip(target, fit.model.lower, fit.model.upper)
    return {'id': identifier, 'q': target.tolist(), 'rail': state['rail'], 'supported': True,
            'duration_s': duration, 'settle_s': settle, 'force_limit_n': 12.}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('state', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--id', required=True)
    parser.add_argument('--operation', choices=['preload', 'balance'], required=True)
    args = parser.parse_args()
    request = prepare(json.loads(args.state.read_text()), args.id, args.operation)
    write_json(args.output/(args.id+'_request.json'), request)
    write_json(args.output/'segment.json', request)


if __name__ == '__main__':
    main()
