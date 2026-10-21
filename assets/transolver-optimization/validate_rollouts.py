"""Compare captured eager/compiled coupled rollouts without changing learned weights."""
import gc
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import warp as wp

from particle_nerd.runner import profile

ROOT = Path('/home/jkamohara/isaac/particle_nerd')
OUT = ROOT / 'experiments/transolver-compile'
parser = argparse.ArgumentParser()
parser.add_argument('--variant', choices=['compiled', 'precision', 'repeat'], default='compiled')
mode = parser.parse_args().variant
if mode != 'compiled':
    OUT = OUT / mode
    OUT.mkdir(exist_ok=True)
original_initialize = profile.initialize_predictor
original_simulation = profile.CoupledSimulation
torch.set_num_threads(1)
args = SimpleNamespace(
    checkpoint=ROOT / 'particle_nerd_plan/results/shovel/training/model/best.pt',
    steps=250, warmup=10, contact_solver='body_velocity',
)
summary = {}
for path in ['wide-circle', 'figure-eight']:
    trajectories = {}
    summary[path] = {}
    for compiled in [False, True]:
        snapshots, poses, wrenches = [], [], []

        def initialize(checkpoint, episode):
            predictor, initial = original_initialize(checkpoint, episode, compile_model=compiled and mode == 'compiled')
            if compiled and mode == 'precision':
                predictor.model = torch.compile(predictor.model, fullgraph=True, dynamic=False,
                    options={'triton.cudagraphs': False, 'emulate_precision_casts': True})
            return predictor, initial

        class RecordingSimulation(original_simulation):
            def diagnostics(self):
                result = super().diagnostics()
                poses.append(result['body_pose'])
                wrenches.append(self.estimator.wrench.numpy().copy())
                if (len(poses) - 1) % 10 == 0:
                    snapshots.append(self.predictor.x.detach().cpu().numpy().copy())
                return result

        profile.initialize_predictor = initialize
        profile.CoupledSimulation = RecordingSimulation
        with torch.inference_mode():
            result = profile.run(
                args, ROOT / f'experiments/particle_gnn/shovel/reference/{path}.hdf5', path, False
            )
        name = mode if compiled else 'eager'
        arrays = dict(particle_positions=np.array(snapshots), body_pose=np.array(poses), wrench=np.array(wrenches))
        trajectories[name] = arrays
        np.savez_compressed(OUT / f'{path}-{name}-rollout.npz', **arrays)
        (OUT / f'{path}-{name}-rollout.json').write_text(json.dumps(result, indent=2) + '\n')
        summary[path][name] = {key: result[key] for key in ['wall', 'iterations_max', 'contacts_range', 'final_pose']}
        print(path, name, summary[path][name], flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    a, b = trajectories['eager'], trajectories[mode]
    particle_error = np.linalg.norm(a['particle_positions'] - b['particle_positions'], axis=-1)
    center_error = np.linalg.norm(a['body_pose'][:, :3] - b['body_pose'][:, :3], axis=-1)
    qa, qb = a['body_pose'][:, 3:], b['body_pose'][:, 3:]
    dot = np.abs(np.sum(qa * qb, axis=-1) / (np.linalg.norm(qa, axis=-1) * np.linalg.norm(qb, axis=-1)))
    angle_error = np.rad2deg(2 * np.arccos(np.clip(dot, 0, 1)))
    force_error = a['wrench'][..., :3] - b['wrench'][..., :3]
    summary[path]['agreement'] = dict(
        particle_position_rms_distance_m=float(np.sqrt(np.mean(particle_error**2))),
        particle_position_max_distance_m=float(particle_error.max()),
        body_center_rms_distance_m=float(np.sqrt(np.mean(center_error**2))),
        body_center_max_distance_m=float(center_error.max()),
        body_orientation_max_degrees=float(angle_error.max()),
        force_rms_component_N=float(np.sqrt(np.mean(force_error**2))),
        reference_force_rms_component_N=float(np.sqrt(np.mean(a['wrench'][..., :3]**2))),
    )
    (OUT / 'rollout-agreement.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(path, 'agreement', summary[path]['agreement'], flush=True)
