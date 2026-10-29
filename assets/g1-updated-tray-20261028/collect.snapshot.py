# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record scheduled native G1/MPM trajectories at every 200 Hz physics step."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from particle_nerd.g1 import DEFAULT_POLICY, MATERIAL_KEYS


def configure(cfg, num_envs):
    """Use a 3 x 3 x 0.3 m fully enclosed flat sand bed and the native 50 Hz policy."""
    cfg.scene.num_envs = num_envs
    cfg.scene.sand.spawn.lower = (-1.5, -1.5, -0.3)
    cfg.scene.sand.spawn.upper = (1.5, 1.5, 0.0)
    cfg.scene.__post_init__()  # Refit native fixed ground and retaining walls.
    cfg.scene.approach_platform = None
    cfg.scene.approach_platform_visual = None
    # Make all four retaining walls (including the -X back wall) and floor form a fully enclosed tray
    wall = 0.1
    wall_height = 0.4  # From z = -0.3 m floor to z = +0.1 m (10 cm above the z = 0 sand surface)
    wall_z = -0.3 + wall_height / 2
    cfg.scene.bed_wall_front.spawn.size = (wall, 3.0 + 2 * wall, wall_height)
    cfg.scene.bed_wall_front.init_state.pos = (1.5 + wall / 2, 0.0, wall_z)
    cfg.scene.bed_wall_front.spawn.visible = True
    cfg.scene.bed_wall_left.spawn.size = (3.0 + 2 * wall, wall, wall_height)
    cfg.scene.bed_wall_left.init_state.pos = (0.0, 1.5 + wall / 2, wall_z)
    cfg.scene.bed_wall_right.spawn.size = (3.0 + 2 * wall, wall, wall_height)
    cfg.scene.bed_wall_right.init_state.pos = (0.0, -(1.5 + wall / 2), wall_z)
    cfg.scene.approach_bank.spawn.size = (wall, 3.0 + 2 * wall, wall_height)
    cfg.scene.approach_bank.init_state.pos = (-(1.5 + wall / 2), 0.0, wall_z)
    cfg.scene.approach_bank.spawn.visible = True
    cfg.scene.robot.init_state.pos = (0.0, 0.0, 0.76)
    cfg.episode_length_s = 6.0
    cfg.video_recorders = []
    cfg.events.reset_base.params["pose_range"] = {k: (0.0, 0.0) for k in ["x", "y", "z", "roll", "pitch", "yaw"]}
    cfg.events.reset_base.params["velocity_range"] = {k: (0.0, 0.0) for k in ["x", "y", "z", "roll", "pitch", "yaw"]}
    cmd = cfg.commands.base_velocity
    cmd.heading_command = False
    cmd.rel_standing_envs = 0.0
    cmd.resampling_time_range = (1000.0, 1000.0)
    cmd.debug_vis = False
    assert cfg.decimation == 4 and np.isclose(cfg.sim.dt, 0.005)
    return cfg


class Writer:
    def __init__(self, path, metadata, static):
        self.path = path
        self.partial = path.with_suffix(".partial.hdf5")
        self.file = h5py.File(self.partial, "x")
        self.file.attrs["manifest"] = json.dumps(metadata, default=str)
        for name, value in static.items():
            self.file.create_dataset(name, data=value)
        self.frames = 0

    def append(self, data):
        for name, value in data.items():
            value = np.asarray(value)
            if not np.isfinite(value).all():
                raise FloatingPointError(f"{self.path}: nonfinite {name}")
            if name not in self.file:
                self.file.create_dataset(
                    name,
                    shape=(0, *value.shape),
                    maxshape=(None, *value.shape),
                    dtype=value.dtype,
                    chunks=(1, *value.shape),
                    compression="lzf",
                    shuffle=True,
                )
            ds = self.file[name]
            ds.resize(len(ds) + 1, axis=0)
            ds[-1] = value
        self.frames += 1

    def close(self, outcome):
        result = dict(outcome=outcome, frames=self.frames, duration_s=(self.frames - 1) * 0.005)
        self.file.attrs["outcome"] = json.dumps(result)
        self.file.flush()
        self.file.close()
        self.partial.rename(self.path)
        result["file_bytes"] = self.path.stat().st_size
        self.path.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
        return result


def select_trials(args):
    trials = json.loads(args.schedule.read_text())["trials"]
    # A pilot subset retains every split and motion group, unlike the first rows.
    if args.limit:
        by_group = {}
        for trial in trials:
            by_group.setdefault((trial["group"], trial["split"]), []).append(trial)
        chosen = []
        while len(chosen) < min(args.limit, len(trials)):
            for group in by_group.values():
                if group and len(chosen) < args.limit:
                    chosen.append(group.pop(0))
        trials = chosen
    trials = trials[args.shard :: args.shards]
    return trials


def main():
    from isaaclab.app import add_launcher_args, launch_simulation

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--schedule", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_POLICY)
    p.add_argument("--num-envs", type=int, default=2)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--limit", type=int)
    p.add_argument("--duration", type=float, default=5.0)
    p.add_argument("--headless", action="store_true")
    add_launcher_args(p)
    args = p.parse_args()
    if not 0 <= args.shard < args.shards or args.num_envs < 1 or args.duration <= 0:
        p.error("Invalid shard/count/duration")
    trials = select_trials(args)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"schedule-shard-{args.shard}.json").write_text(json.dumps(dict(trials=trials), indent=2) + "\n")
    trials = [t for t in trials if not (args.output / f"episode-{t['trial_id']:04d}.hdf5").exists()]
    if not trials:
        print("COLLECTION_COMPLETE no pending trials", flush=True)
        return
    torch.set_num_threads(2)
    import gymnasium as gym
    import newton
    from isaaclab_newton.physics import NewtonManager
    from rsl_rl.models import MLPModel

    from isaaclab.utils.math import quat_from_euler_xyz

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    task = "IsaacContrib-Velocity-Sand-G1-29dof-MPM-Play"
    cfg = configure(parse_env_cfg(task, num_envs=args.num_envs), args.num_envs)
    cfg.episode_length_s = args.duration + 1.0
    cfg.seed = 120000 + args.shard
    saved = json.loads(args.checkpoint.with_name("wandb_config.json").read_text())
    launcher = dict(vars(args), visualizer=[] if args.headless else (args.visualizer or ["newton_rtx"]))
    for visualizer in cfg.sim.visualizer_cfgs:
        visualizer.headless = args.headless
    with launch_simulation(cfg, launcher):
        env = RslRlVecEnvWrapper(gym.make(task, cfg=cfg), clip_actions=saved["train_cfg"].get("clip_actions"))
        raw = env.unwrapped
        opened = []
        hook_original = raw.recorder_manager.record_post_physics_decimation_step
        try:
            obs = env.get_observations()
            actor_cfg = saved["train_cfg"]["actor"].copy()
            assert actor_cfg.pop("class_name") == "MLPModel"
            actor = MLPModel(
                obs, saved["train_cfg"]["obs_groups"], "actor", raw.action_manager.total_action_dim, **actor_cfg
            ).to(raw.device)
            actor.load_state_dict(
                torch.load(args.checkpoint, map_location=raw.device, weights_only=True)["actor_state_dict"]
            )
            actor.eval()
            # Preserve four actuator/physics updates per policy action while exposing
            # each completed 5 ms state to the recorder outside CUDA graph capture.
            NewtonManager.set_decimation(1)
            raw._physics_handles_decimation = False
            model = NewtonManager.get_model()
            pw, bw = model.particle_world.numpy(), model.body_world.numpy()
            pids = [np.flatnonzero(pw == e) for e in range(args.num_envs)]
            bids = [np.flatnonzero(bw == e) for e in range(args.num_envs)]
            assert len(set(map(len, pids))) == 1
            jw = model.joint_world.numpy()
            qstart, vstart = model.joint_q_start.numpy(), model.joint_qd_start.numpy()
            qids = [
                np.concatenate([np.arange(qstart[j], qstart[j + 1]) for j in np.flatnonzero(jw == e)])
                for e in range(args.num_envs)
            ]
            vids = [
                np.concatenate([np.arange(vstart[j], vstart[j + 1]) for j in np.flatnonzero(jw == e)])
                for e in range(args.num_envs)
            ]
            origins = raw.scene.env_origins.cpu().numpy()
            feet = raw._foot_body_ids.cpu().numpy()
            sb, st = model.shape_body.numpy(), model.shape_type.numpy()
            labels = list(model.body_label)
            shapes = []
            for e in range(args.num_envs):
                eligible = set(feet[e]) | {b for b in bids[e] if "/MPM" in labels[b]}
                ids = np.array([i for i, b in enumerate(sb) if b in eligible and st[i] == newton.GeoType.BOX])
                if len(ids) < 7:
                    raise ValueError(f"Expected both feet and five native boundary boxes, found {len(ids)}")
                shapes.append(ids)
            print(
                "COLLECTION_SETUP",
                json.dumps(
                    dict(
                        particles_per_environment=len(pids[0]),
                        collider_counts=list(map(len, shapes)),
                        environments=args.num_envs,
                        physics_hz=200,
                        policy_hz=50,
                        terrain_m=[3, 3, 0.3],
                    )
                ),
                flush=True,
            )
            policy_sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
            for start in range(0, len(trials), args.num_envs):
                batch = trials[start : start + args.num_envs]
                torch.manual_seed(120000 + args.shard * 10000 + start)
                env.reset()
                # Assign planned world-local base poses after native sand/joint reset.
                robot = raw.scene["robot"]
                pose = robot.data.root_state_w.torch.clone()
                for e, trial in enumerate(batch):
                    pose[e, :3] = torch.tensor(
                        origins[e] + [trial["initial_x_m"], trial["initial_y_m"], 0.76], device=raw.device
                    )
                    yaw = torch.tensor([trial["initial_yaw_rad"]], device=raw.device)
                    pose[e, 3:7] = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)[0]
                    pose[e, 7:] = 0
                robot.write_root_state_to_sim(pose)
                raw.scene.write_data_to_sim()
                raw.sim.forward()
                raw.scene.update(dt=0.0)
                obs = raw.observation_manager.compute()
                active = [True] * len(batch)
                opened = []
                for e, trial in enumerate(batch):
                    ids = shapes[e]
                    metadata = dict(
                        schema="g1_particle_nerd_v1",
                        trial=trial,
                        split=trial["split"],
                        physics_dt_s=0.005,
                        policy_dt_s=0.02,
                        record_hz=200,
                        requested_duration_s=args.duration,
                        material={k: getattr(cfg.scene.sand.spawn.material, k) for k in MATERIAL_KEYS},
                        bed_dimensions_m=[3, 3, 0.3],
                        grid_spacing_m=0.04,
                        policy_sha256=policy_sha,
                        coordinates="Positions relative to environment origin; world-axis vectors; quaternion xyzw",
                        velocity_reference="body_velocity linear part is at body COM; body_pose is body frame",
                        generalized_force_note="Control.joint_f; excludes native actuator output",
                        foot_wrench_timing="Material feedback for this 5 ms step; no policy-interval averaging",
                        reset_rule="Record pre-reset physics states; stop writing this world after native termination",
                        runtime_seed=120000 + args.shard * 10000 + start,
                        config=cfg.to_dict(),
                    )
                    local_body = {int(b): i for i, b in enumerate(bids[e])}
                    static = dict(
                        particle_mass=model.particle_mass.numpy()[pids[e]],
                        particle_radius=model.particle_radius.numpy()[pids[e]],
                        body_com_local=model.body_com.numpy()[bids[e]],
                        collider_body=np.array([local_body[int(sb[i])] for i in ids]),
                        collider_local_pose=model.shape_transform.numpy()[ids],
                        collider_half_size=model.shape_scale.numpy()[ids],
                        collider_friction=model.shape_material_mu.numpy()[ids],
                        foot_body_ids=np.array([local_body[int(b)] for b in feet[e]]),
                    )
                    opened.append(Writer(args.output / f"episode-{trial['trial_id']:04d}.hdf5", metadata, static))
                step = 0
                current_action = torch.zeros(args.num_envs, raw.action_manager.total_action_dim, device=raw.device)
                current_command = torch.zeros(args.num_envs, 3, device=raw.device)
                noise = torch.zeros_like(current_action)
                target = torch.zeros_like(current_action)

                def record():
                    state = NewtonManager.get_state_0()
                    xp, vp, bq, bv = (
                        state.particle_q.numpy(),
                        state.particle_qd.numpy(),
                        state.body_q.numpy(),
                        state.body_qd.numpy(),
                    )
                    force = raw._coupling_forces.numpy()
                    jq, jv = state.joint_q.numpy(), state.joint_qd.numpy()
                    jf = NewtonManager.get_control().joint_f.numpy()
                    action_np, command_np = current_action.cpu().numpy(), current_command.cpu().numpy()
                    for e, writer in enumerate(opened):
                        if active[e]:
                            poses = bq[bids[e]].copy()
                            poses[:, :3] -= origins[e]
                            writer.append(
                                dict(
                                    time_s=np.float64(step * 0.005),
                                    particle_position=xp[pids[e]] - origins[e],
                                    particle_velocity=vp[pids[e]],
                                    body_pose=poses,
                                    body_velocity=bv[bids[e]],
                                    foot_material_wrench=force[feet[e]],
                                    action=action_np[e],
                                    joint_position=jq[qids[e]],
                                    joint_velocity=jv[vids[e]],
                                    applied_generalized_force=jf[vids[e]],
                                    command=command_np[e],
                                )
                            )

                def after_physics():
                    nonlocal step
                    hook_original()
                    step += 1
                    record()

                raw.recorder_manager.record_post_physics_decimation_step = after_physics
                record()
                term = raw.command_manager.get_term("base_velocity")
                wall_start = time.perf_counter()
                with torch.no_grad():
                    for k in range(round(args.duration / 0.02)):
                        for e, trial in enumerate(batch):
                            command = (
                                trial["command_end"] if k * 0.02 >= trial["switch_time_s"] else trial["command_start"]
                            )
                            current_command[e] = torch.tensor(command, device=raw.device)
                        term.vel_command_b[:] = current_command
                        term.is_standing_env[:] = False
                        obs["command"][:] = term.command
                        current_action = actor(obs)
                        if k % 10 == 0:
                            target.uniform_(-1, 1)
                        noise.lerp_(target, 0.2)
                        for e, trial in enumerate(batch):
                            amplitude = trial["action_noise_amplitude"]
                            if trial["mode"] == "policy-noise":
                                current_action[e] += amplitude * noise[e]
                            elif trial["mode"] == "random":
                                current_action[e] = amplitude * noise[e]
                            elif trial["mode"] == "zero":
                                current_action[e].zero_()
                        obs, _, done, _ = env.step(current_action)
                        for e, writer in enumerate(opened):
                            if active[e] and bool(done[e]):
                                report = writer.close("native_termination")
                                active[e] = False
                                print(
                                    "EPISODE_RESULT",
                                    json.dumps(dict(trial_id=batch[e]["trial_id"], **report)),
                                    flush=True,
                                )
                        if not any(active):
                            break
                raw.recorder_manager.record_post_physics_decimation_step = hook_original
                for e, writer in enumerate(opened):
                    if active[e]:
                        assert writer.frames == 1 + round(args.duration / 0.005), (writer.frames, args.duration)
                        report = writer.close("duration")
                        active[e] = False
                        print("EPISODE_RESULT", json.dumps(dict(trial_id=batch[e]["trial_id"], **report)), flush=True)
                print(
                    "BATCH_COMPLETE",
                    json.dumps(dict(wall_seconds=time.perf_counter() - wall_start, physics_steps=step)),
                    flush=True,
                )
                opened = []
            print("COLLECTION_COMPLETE", flush=True)
        finally:
            raw.recorder_manager.record_post_physics_decimation_step = hook_original
            for writer in opened:
                if writer.file.id.valid:
                    writer.file.close()
            env.close()


if __name__ == "__main__":
    main()
