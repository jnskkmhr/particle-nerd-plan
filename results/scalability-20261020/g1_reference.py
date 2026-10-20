# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Preview or record the native dynamic G1 + MPM task for particle-motion learning.

Use the current IsaacLab checkout's runtime/task implementation. This script does
not replace particle dynamics or robot feedback with a learned model. Fifty-Hz
records are transitions; reset transitions are explicitly invalidated.
"""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import h5py
import numpy as np
import torch

DEFAULT_POLICY = Path(
    "/home/jkamohara/isaac/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/contrib/g1_29dof_mpm/benchmark/data/checkpoints/5f1xyc21/model_39998.pt"
)
MATERIAL_KEYS = (
    "density",
    "young_modulus",
    "poisson_ratio",
    "friction",
    "yield_pressure",
    "tensile_yield_ratio",
    "yield_stress",
    "hardening",
    "dilatancy",
    "viscosity",
)


def main():
    from isaaclab.app import add_launcher_args, launch_simulation

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="IsaacContrib-Velocity-Sand-G1-29dof-MPM-Play")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--mode", choices=["policy", "policy-noise", "zero", "random"], default="policy")
    parser.add_argument("--command", type=float, nargs=3, default=(0.15, 0.0, 0.0), metavar=("VX", "VY", "WZ"))
    parser.add_argument("--command-end", type=float, nargs=3, metavar=("VX", "VY", "WZ"))
    parser.add_argument("--switch-time", type=float, default=2.5)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--noise-amplitude", type=float, default=0.1, help="Bounded normalized-action perturbation amplitude"
    )
    parser.add_argument(
        "--material-friction", type=float, help="Override sand internal friction only; rigid contact friction unchanged"
    )
    parser.add_argument("--output", type=Path, help="HDF5 output; omit to preview without recording")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--video-dir", type=Path, help="Record the native Newton RTX view to this directory")
    add_launcher_args(parser)
    args = parser.parse_args()
    if args.duration <= 0 or args.num_envs < 1 or args.noise_amplitude < 0:
        parser.error("Invalid duration/count/noise")
    if args.material_friction is not None and args.material_friction < 0:
        parser.error("Friction must be nonnegative")
    if args.output and args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    import gymnasium as gym
    from isaaclab_newton.physics import NewtonManager
    from rsl_rl.models import MLPModel

    from isaaclab.paths import ISAACLAB_ROOT

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    cfg = parse_env_cfg(args.task, num_envs=args.num_envs)
    cfg.seed = args.seed
    cfg.video_recorders = []
    for visualizer in cfg.sim.visualizer_cfgs:
        visualizer.headless = args.headless
    if args.video_dir:
        from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg

        cfg.video_recorders = [
            VideoRecorderCfg(
                source="viz:newton_rtx",
                output_dir=str(args.video_dir),
                fps=50,
                video_length=math.ceil(args.duration / 0.02),
                output_filename_prefix="g1-reference",
            )
        ]
    cfg.episode_length_s = args.duration + 1.0
    # Start on material so five seconds contain foot/sand interaction, rather
    # than spending most of the recording crossing the rigid approach platform.
    cfg.scene.robot.init_state.pos = (0.0, 0.0, 0.76)
    cfg.events.reset_base.params["pose_range"].update(x=(-0.05, 0.05), y=(-0.05, 0.05))
    cmd = cfg.commands.base_velocity
    cmd.ranges.lin_vel_x = (args.command[0], args.command[0])
    cmd.ranges.lin_vel_y = (args.command[1], args.command[1])
    cmd.ranges.ang_vel_z = (args.command[2], args.command[2])
    cmd.heading_command = False
    cmd.rel_standing_envs = 0.0
    cmd.resampling_time_range = (1000.0, 1000.0)
    cmd.debug_vis = False
    if args.material_friction is not None:
        cfg.scene.sand.spawn.material.friction = args.material_friction
    # Native physics, contacts, actuation, terrain size and display are retained.
    saved = (
        json.loads(args.checkpoint.with_name("wandb_config.json").read_text())
        if args.mode.startswith("policy")
        else None
    )
    launcher = dict(vars(args), visualizer=[] if args.headless else (args.visualizer or ["newton_rtx"]))
    partial = None
    h5 = None
    with launch_simulation(cfg, launcher):
        env = RslRlVecEnvWrapper(
            gym.make(args.task, cfg=cfg), clip_actions=None if saved is None else saved["train_cfg"].get("clip_actions")
        )
        raw = env.unwrapped
        try:
            obs = env.get_observations()
            if not np.isclose(raw.step_dt, 0.02):
                raise ValueError("This recorder expects the native 50 Hz policy interval")
            actor = None
            if saved:
                actor_cfg = saved["train_cfg"]["actor"].copy()
                if actor_cfg.pop("class_name") != "MLPModel":
                    raise ValueError("Unsupported policy architecture")
                actor = MLPModel(
                    obs, saved["train_cfg"]["obs_groups"], "actor", raw.action_manager.total_action_dim, **actor_cfg
                ).to(raw.device)
                actor.load_state_dict(
                    torch.load(args.checkpoint, map_location=raw.device, weights_only=True)["actor_state_dict"],
                    strict=True,
                )
                actor.eval()
            model = NewtonManager.get_model()
            particle_world = model.particle_world.numpy()
            body_world = model.body_world.numpy()
            particle_ids = np.stack([np.flatnonzero(particle_world == e) for e in range(args.num_envs)])
            body_ids = np.stack([np.flatnonzero(body_world == e) for e in range(args.num_envs)])
            origins = raw.scene.env_origins.cpu().numpy()
            feet = raw._foot_body_ids.cpu().numpy()
            metadata = dict(
                task=args.task,
                isaaclab_root=str(ISAACLAB_ROOT),
                mode=args.mode,
                seed=args.seed,
                environments=args.num_envs,
                particles_per_environment=particle_ids.shape[1],
                record_hz=50,
                physics_dt_s=cfg.sim.dt,
                requested_duration_s=args.duration,
                command_start=args.command,
                command_end=args.command_end,
                switch_time_s=args.switch_time,
                noise_amplitude=args.noise_amplitude,
                material={k: getattr(cfg.scene.sand.spawn.material, k) for k in MATERIAL_KEYS},
                bed_dimensions_m=(
                    np.asarray(cfg.scene.sand.spawn.upper) - np.asarray(cfg.scene.sand.spawn.lower)
                ).tolist(),
                grid_spacing_m=cfg.scene.sand.spawn.voxel_size,
                checkpoint=str(args.checkpoint) if saved else None,
                checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() if saved else None,
                coordinates=(
                    "Particle and body positions relative to environment origin; vectors "
                    "in world axes; body quaternion xyzw"
                ),
                wrench_order="World force xyz [N], world torque xyz [N m]; native MPM coupling feedback",
                frame_timing=(
                    "Initial frame plus states after each 20 ms policy interval; action[k] produces frame[k+1]"
                ),
                force_timing=(
                    "foot_material_wrench is newly generated coupling feedback from the "
                    "last 5 ms substep, not a 20 ms average"
                ),
                action_note=(
                    "zero means zero normalized joint-position action, not zero actuator "
                    "torque; random is temporally smoothed"
                ),
                reset_rule=(
                    "Stop the batch on the first native termination; transition_valid is false for reset environments"
                ),
                geometry_note="Native flat testbed retained. Robot starts at center of sand; no approach traversal",
                config=cfg.to_dict(),
            )
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                partial = args.output.with_suffix(".partial.hdf5")
                h5 = h5py.File(partial, "x")
                h5.attrs["manifest"] = json.dumps(metadata, default=str)
                h5.create_dataset("particle_mass", data=model.particle_mass.numpy()[particle_ids])
                h5.create_dataset("particle_radius", data=model.particle_radius.numpy()[particle_ids])
                h5.create_dataset("particle_global_ids", data=particle_ids)
                h5.create_dataset("body_global_ids", data=body_ids)
                h5.create_dataset("body_com_local", data=model.body_com.numpy()[body_ids])
                h5.create_dataset("environment_origins", data=origins)
                h5.create_dataset("foot_body_global_ids", data=feet)
                h5.attrs["body_labels"] = json.dumps(list(model.body_label))
                for key in ["shape_body", "shape_type", "shape_transform", "shape_scale", "shape_material_mu"]:
                    h5.create_dataset("geometry/" + key, data=getattr(model, key).numpy())

            def append(name, value):
                if h5 is None:
                    return
                value = np.asarray(value)
                if name not in h5:
                    h5.create_dataset(
                        name,
                        shape=(0, *value.shape),
                        maxshape=(None, *value.shape),
                        dtype=value.dtype,
                        chunks=(1, *value.shape),
                        compression="lzf",
                    )
                ds = h5[name]
                ds.resize(len(ds) + 1, axis=0)
                ds[-1] = value

            def record(t):
                state = NewtonManager.get_state_0()
                position = state.particle_q.numpy()[particle_ids] - origins[:, None, :]
                velocity = state.particle_qd.numpy()[particle_ids]
                pose = state.body_q.numpy()[body_ids]
                pose[:, :, :3] -= origins[:, None, :]
                if not np.isfinite(position).all() or not np.isfinite(velocity).all() or not np.isfinite(pose).all():
                    raise FloatingPointError("Nonfinite reference state")
                append("time_s", np.float64(t))
                append("particle_position", position)
                append("particle_velocity", velocity)
                append("body_pose", pose)
                append("body_velocity", state.body_qd.numpy()[body_ids])
                append("joint_position", raw.scene["robot"].data.joint_pos.torch.cpu().numpy())
                append("joint_velocity", raw.scene["robot"].data.joint_vel.torch.cpu().numpy())
                append("foot_material_wrench", raw._coupling_forces.numpy()[feet])

            print(
                "REFERENCE_SETUP",
                json.dumps({k: v for k, v in metadata.items() if k != "config"}, default=str),
                flush=True,
            )
            record(0.0)
            term = raw.command_manager.get_term("base_velocity")
            action = torch.zeros(args.num_envs, raw.action_manager.total_action_dim, device=raw.device)
            noise = torch.zeros_like(action)
            target_noise = torch.zeros_like(action)
            outcome = "duration"
            completed = 0
            with torch.inference_mode():
                for k in range(math.ceil(args.duration / raw.step_dt)):
                    started = time.perf_counter()
                    t = k * raw.step_dt
                    command = (
                        args.command_end if args.command_end is not None and t >= args.switch_time else args.command
                    )
                    term.vel_command_b[:] = torch.tensor(command, device=raw.device)
                    term.is_standing_env[:] = False
                    obs["command"][:] = term.command
                    if k % 10 == 0:
                        target_noise.uniform_(-args.noise_amplitude, args.noise_amplitude)
                    noise.lerp_(target_noise, 0.2)
                    if actor is not None:
                        action = actor(obs)
                        if args.mode == "policy-noise":
                            action = action + noise
                    elif args.mode == "random":
                        action = noise
                    else:
                        action.zero_()
                    if not torch.isfinite(action).all():
                        raise FloatingPointError("Nonfinite action")
                    append("action", action.cpu().numpy())
                    append("command", term.command.cpu().numpy())
                    obs, _, done, _ = env.step(action)
                    completed = k + 1
                    append("transition_valid", (~done.bool()).cpu().numpy())
                    record(completed * raw.step_dt)
                    if bool(done.any()):
                        outcome = "native_termination"
                        break
                    if not args.headless:
                        time.sleep(max(0.0, raw.step_dt - (time.perf_counter() - started)))
            report = dict(
                outcome=outcome,
                completed_steps=completed,
                recorded_duration_s=completed * raw.step_dt,
                particles_per_environment=particle_ids.shape[1],
                environments=args.num_envs,
            )
            if h5:
                h5.attrs["outcome"] = json.dumps(report)
                h5.flush()
                h5.close()
                h5 = None
                partial.rename(args.output)
                report["file_bytes"] = args.output.stat().st_size
                args.output.with_suffix(".json").write_text(
                    json.dumps(dict(metadata=metadata, result=report), indent=2, default=str) + "\n"
                )
            print("REFERENCE_RESULT", json.dumps(report), flush=True)
        finally:
            if h5 is not None:
                h5.close()
            env.close()


if __name__ == "__main__":
    main()
