"""Fit a demonstrated wrap around a rail with fixed world orientation."""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[2])]
from bottom_grasp import BottomGraspFit
from common import write_json
from four_finger_geometry import FourFingerGeometry

from bimanual_teleop.sim.config import ROOT
from bimanual_teleop.sim.kinematics import RobotModel, pose_vector


class DemonstrationFit(BottomGraspFit):
    def __init__(self, model, assets, state, reference):
        super().__init__(model, assets/'robot.urdf', assets/'manifest.json', state)
        self.reference = reference
        self.base_correction = self.correction.copy()
        self.link_corrections = {}  # Planning uses URDF; measured refinement follows.
        self.flange = self.correction @ model.flange('left', self.q, world=True)
        self.world_rail = self.rail_pose.copy()
        self.reference_rotation = np.array(reference['T_W_F'])[:3, :3]
        self.cache_x = None
        self.cache = None

    def unpack(self, x):
        q = self.q.copy()
        q[14:34] = x[:20]
        target = self.flange.copy()
        target[:3, :3] = Rotation.from_rotvec(x[20:23]).as_matrix() @ self.flange[:3, :3]
        self.correction = target @ np.linalg.inv(self.model.flange('left', self.q, world=True))
        xyz = self.world_rail[:3, 3] + .01*x[23:26]
        return q, xyz

    def evaluate_all(self, x):
        if self.cache_x is None or not np.array_equal(x, self.cache_x):
            q, xyz = self.unpack(x)
            ds = self.distances(q, xyz)
            sd = self.self_distances()
            bm = self.bottom_metrics(q, xyz)
            self.cache = ds, sd, bm
            self.cache_x = x.copy()
        return self.cache

    def solve_reference(self, output):
        x0 = np.r_[self.q[14:34], np.zeros(6)]
        target = np.asarray(self.reference['measured_position_rad'])
        x0[:20] = target
        x0[20:23] = Rotation.from_matrix(self.reference_rotation @ self.flange[:3, :3].T).as_rotvec()
        initial_q, _ = self.unpack(x0)
        poses = self.transforms(initial_q)
        center = np.mean([poses['l_'+f+'_tip_sensor_frame'][:3, 3] for f in self.pad_vertices], axis=0)
        x0[23:26] = (center + self.world_rail[:3, 2]*.016 - self.world_rail[:3, 3])/.01
        def objective(x):
            ds, _, bm = self.evaluate_all(x)
            qerr = x[:20]-target
            rotation = Rotation.from_rotvec(x[20:23]).as_matrix() @ self.flange[:3, :3]
            rerr = Rotation.from_matrix(rotation.T@self.reference_rotation).as_rotvec()
            value = qerr@qerr + 2.*(rerr@rerr) + .0003*(x[23:]@x[23:])
            value += 2.*((ds[self.names.index('l_wrist')]-.00025)/.01)**2
            value += sum(((v['gap_m']-.00025)/.01)**2 for v in bm.values())
            value += ((ds[self.tips[0]]-.00025)/.01)**2
            return value
        def constraints(x):
            ds, sd, bm = self.evaluate_all(x)
            values = list((ds-.00012)/.01) + list((sd-.0003)/.01)
            values += [(.0006-ds[self.tips[0]])/.01]
            values += [(.0006-ds[self.names.index('l_wrist')])/.01]
            for v in bm.values():
                values += [(v['gap_m']-.00015)/.01, (.0006-v['gap_m'])/.01,
                           v['inward_alignment']-.94, (.009-abs(v['rail_point_m'][1]))/.01,
                           (-.004-v['pad_point_m'][1])/.01]
            return np.asarray(values)
        bounds = list(zip(self.model.lower[14:34]+1e-6, self.model.upper[14:34]-1e-6))
        bounds += [(-.7, .7)]*3 + [(-5., 5.)]*3
        def residual(x):
            ds, _, bm = self.evaluate_all(x)
            rotation = Rotation.from_rotvec(x[20:23]).as_matrix() @ self.flange[:3, :3]
            rerr = Rotation.from_matrix(rotation.T@self.reference_rotation).as_rotvec()
            contact = [ds[self.names.index('l_wrist')]-.00025, ds[self.tips[0]]-.00025]
            contact += [v['gap_m']-.00025 for v in bm.values()]
            return np.r_[x[:20]-target, 1.4*rerr, .02*x[23:],
                         100*np.asarray(contact), 100*np.minimum(constraints(x), 0.)]
        preliminary = least_squares(residual, np.clip(x0, np.array(bounds)[:, 0]+1e-7, np.array(bounds)[:, 1]-1e-7),
                                    bounds=np.array(bounds).T, max_nfev=250, diff_step=1e-4,
                                    jac='3-point', x_scale='jac', ftol=1e-10, xtol=1e-10, gtol=1e-8)
        x0 = preliminary.x
        print('preliminary', preliminary.cost, constraints(x0).min(), flush=True)
        result = SimpleNamespace(x=x0, success=constraints(x0).min() > -.001,
                                 message='Bounded least-squares contact fit; full mesh acceptance required', fun=objective(x0))
        q, xyz = self.unpack(result.x)
        target_flange = self.correction @ self.model.flange('left', q, world=True)
        # Keep the current flange position, change its orientation, and solve
        # the real seven arm joints before auditing the fitted hand.
        def ik(values):
            trial = q.copy(); trial[:7] = values
            actual = self.model.flange('left', trial, world=True)
            return np.r_[30*(actual[:3, 3]-target_flange[:3, 3]),
                         Rotation.from_matrix(target_flange[:3, :3].T@actual[:3, :3]).as_rotvec(),
                         .0001*(values-self.q[:7])]
        solved = least_squares(ik, q[:7], bounds=(self.model.lower[:7], self.model.upper[:7]), max_nfev=300,
                               ftol=1e-12, xtol=1e-12, gtol=1e-12)
        q[:7] = solved.x
        rail = self.world_rail.copy(); rail[:3, 3] = xyz
        check_state = {'q': q.tolist(), 'eef': [pose_vector(self.model.flange('left', q)).tolist()],
                       'rail': pose_vector(rail).tolist(), 'time_ns': 0}
        geometry = FourFingerGeometry(self.model, ROOT/'assets/generated/robot.urdf', ROOT/'assets/generated/manifest.json').check(check_state)
        ds, sd, bm = self.evaluate_all(result.x)
        plan = {'source_episode': self.reference['source_episode'], 'q': q.tolist(), 'rail': pose_vector(rail).tolist(),
                'hand_measured_reference_rad': target.tolist(), 'hand_adjustment_deg': np.rad2deg(q[14:34]-target).reshape(5, 4).tolist(),
                'flange_pose_world': pose_vector(target_flange).tolist(), 'optimizer_x': result.x.tolist(),
                'solver_success': bool(result.success), 'solver_message': str(result.message),
                'minimum_constraint': float(constraints(result.x).min()), 'objective': float(result.fun),
                'minimum_envelope_gap_m': float(ds.min()), 'minimum_interfinger_gap_m': float(sd.min()),
                'palm_gap_m': float(ds[self.names.index('l_wrist')]), 'bottom': bm, 'geometry': geometry,
                'arm_ik_residual': ik(q[:7])[:6].tolist(), 'rail_orientation_preserved': True,
                'collision_settings_unchanged': True, 'physical_grip_validated': False}
        write_json(output/'plan.json', plan)
        return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--state', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    state = json.loads(args.state.read_text()); state = state.get('state', state)
    reference = json.loads((HERE/'hand_profiles/recorded_left_tight_grasp.json').read_text())
    assets = ROOT/'assets/generated'
    plan = DemonstrationFit(RobotModel(assets/'robot.urdf'), assets, state, reference).solve_reference(args.output)
    print(json.dumps({k: plan[k] for k in ('solver_success', 'solver_message', 'minimum_constraint', 'palm_gap_m', 'hand_adjustment_deg', 'arm_ik_residual')}, indent=2))


if __name__ == '__main__':
    main()
