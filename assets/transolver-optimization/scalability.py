# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment-count benchmark with fixed per-environment particles and timestep.

Run each backend/count in a separate process. Learned neural evaluation is truly
batched; existing single-body contact estimators and MJWarp instances are replayed
per environment. This explicitly measures that implementation's scaling limit.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import warp as wp

ROOT = Path("/home/jkamohara/isaac/particle_nerd")


def summary(values):
    return dict(
        median_ms=float(np.median(values)), p95_ms=float(np.percentile(values, 95)), mean_ms=float(np.mean(values))
    )


def memory():
    free, total = torch.cuda.mem_get_info()
    return dict(
        device_used_GiB=(total - free) / 2**30, torch_peak_allocated_GiB=torch.cuda.max_memory_allocated() / 2**30
    )


def measure(operation, steps, warmup):
    for k in range(warmup):
        operation(k)
        torch.cuda.synchronize()
    torch.cuda.synchronize()
    gpu, wall = [], []
    start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for k in range(warmup, warmup + steps):
        torch.cuda.synchronize()
        start = time.perf_counter()
        start_event.record()
        operation(k)
        end_event.record()
        end_event.synchronize()
        wall.append((time.perf_counter() - start) * 1000)
        gpu.append(start_event.elapsed_time(end_event))
    return dict(wall=summary(wall), gpu_stream_interval=summary(gpu), raw_wall_ms=wall)


def make_inputs(predictor, pose, velocity):
    from particle_nerd.transolver.features import features
    from particle_nerd.transolver.inference import box_ground_state

    poses, velocities = box_ground_state(pose, velocity)
    return features(
        predictor.history,
        poses,
        velocities,
        predictor.kinds,
        predictor.sizes,
        predictor.friction,
        predictor.material,
        predictor.dt,
        **predictor.config["features"],
        sdf=predictor.sdf,
    )


def capture_network(model, inputs):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        for _ in range(3):
            model(*inputs)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = model(*inputs).contiguous()
    torch.cuda.current_stream().wait_stream(stream)
    return graph, output


def learned(args):
    from particle_nerd.data_collection.dataset import initialize_predictor
    from particle_nerd.gnn_v2.geometry import update
    from particle_nerd.runner.simulation import CoupledSimulation

    predictor, initial = initialize_predictor(args.checkpoint, args.episode, compile_model=args.compile)
    inputs = make_inputs(
        predictor,
        torch.as_tensor(initial["body_pose"], device="cuda"),
        torch.as_tensor(initial["body_velocity"], device="cuda"),
    )
    batch = tuple(x.unsqueeze(0).repeat(args.envs, *([1] * x.ndim)).contiguous() for x in inputs)
    graph, displacement = capture_network(predictor.model, batch)
    common = dict(
        particles_per_environment=len(predictor.x),
        neural_config=predictor.config,
        neural_precision="BF16 autocast",
        neural_cuda_graph=True,
        neural_compiled=args.compile,
    )
    if args.backend == "network":
        result = measure(lambda k: graph.replay(), args.steps, args.warmup)
        return dict(
            **common,
            timings=result,
            finite=bool(torch.isfinite(displacement).all()),
            scope="Neural forward only; fixed recorded inputs; excludes geometry, updates, forces and rigid dynamics",
        )
    simulations = []
    for e in range(args.envs):
        p, state = (predictor, initial) if e == 0 else initialize_predictor(args.checkpoint, args.episode)
        p.model = predictor.model  # One checkpoint/weight allocation for all worlds.

        # The coordinator supplies the already computed displacement. The regular
        # CoupledSimulation retains its original ordering: save old x, update
        # particles, estimate force, then four rigid substeps.
        def staged_step(body_pose, body_velocity, p=p, e=e):
            wp.launch(
                update,
                len(p.x),
                [
                    wp.from_torch(p.history),
                    wp.from_torch(p.x, dtype=wp.vec3),
                    wp.from_torch(p.v, dtype=wp.vec3),
                    wp.from_torch(displacement[e]),
                    p.active,
                    p.dt,
                ],
                device=p.device,
            )
            return p.x, p.v

        p.step = staged_step
        simulation = CoupledSimulation(
            p,
            state["manifest"],
            state["particle_mass"],
            state["body_pose"],
            state["body_velocity"],
            effective_mass_scale=180,
            jacobi_relaxation=0.075,
            max_iterations=2000,
            contact_solver="body_velocity",
            use_cuda_graph=True,
        )
        simulations.append(simulation)
    stage_rows = []
    events = []
    instrument = False

    def stage(name, fn):
        if not instrument:
            return fn()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        events.append((name, a, b))
        return None

    def operation(k):
        # Every world has its own state. No particle attention spans worlds.
        def geometry():
            for e, sim in enumerate(simulations):
                torch.cuda.current_stream().wait_stream(sim.torch_stream)
                values = make_inputs(
                    sim.predictor, wp.to_torch(sim.rigid.state.body_q)[0], wp.to_torch(sim.rigid.state.body_qd)[0]
                )
                for target, value in zip(batch, values, strict=True):
                    target[e].copy_(value)

        stage("features_and_batch_copy", geometry)
        stage("neural_network", graph.replay)

        def remainder():
            for sim in simulations:
                sim.step()
                torch.cuda.current_stream().wait_stream(sim.torch_stream)

        stage("particle_update_force_and_rigid", remainder)
        # Event values are collected on the next iteration, after measure's sync.
        if len(events) > 3:
            previous = events[:-3]
            stage_rows.append({name: a.elapsed_time(b) for name, a, b in previous})
            del events[:-3]

    result = measure(operation, args.steps, args.warmup)
    instrument = True
    measure(operation, 10, 2)
    stage_rows.append({name: a.elapsed_time(b) for name, a, b in events})
    measured = stage_rows[2:]
    diagnostics = [sim.diagnostics() for sim in simulations]
    valid = all(
        d["particle"]["finite"]
        and d["particle"]["velocity_finite"]
        and d["contact"]["finite"]
        and d["contact"]["converged"]
        and not d["rigid_contact"]["overflow"]
        for d in diagnostics
    )
    return dict(
        **common,
        timings=result,
        finite=valid,
        stage_scope=(
            "Separate instrumented 10-step continuation after primary timings; eager features, captured NN/force/rigid"
        ),
        stages={key: summary([r[key] for r in measured]) for key in measured[0]},
        final_contacts=[d["contact"]["contacts"] for d in diagnostics],
        final_iterations=[d["contact"]["iterations"] for d in diagnostics],
        scope="Batched NN + per-environment GPU features, particle update, captured force estimator and MJWarp",
        limitation=(
            "Force and rigid solvers remain per-environment instances, dispatched "
            "in a Python loop; not a fused batched contact solver"
        ),
    )


def mpm(args):
    from isaaclab_newton.physics import NewtonManager
    from isaaclab_newton.sim.spawners.mpm import MPMPointsCfg
    from particle_nerd.data_collection.collect import create_collection_env_cfg
    from particle_nerd.data_collection.dataset import load_initial_state

    from isaaclab.app import launch_simulation
    from isaaclab.envs import ManagerBasedEnv

    initial = load_initial_state(args.episode)
    m = initial["manifest"]
    n = len(initial["position"])
    cfg, _ = create_collection_env_cfg("shovel", m["body_mass_kg"], 42, bed_size=m["scene"]["bed_size_m"])
    cfg.scene.num_envs = args.envs
    material = cfg.scene.sand.spawn.material
    for key, value in m["material"].items():
        setattr(material, key, value)
    cfg.scene.sand.spawn = MPMPointsCfg(
        positions=initial["position"].tolist(),
        velocities=initial["velocity"].tolist(),
        mass=initial["particle_mass"].tolist(),
        radius=initial["particle_radius"].tolist(),
        material=material,
    )
    cfg.events.reset_material.params["jitter_fraction"] = 0.0
    cfg.events.randomize_box = None
    cfg.sim.physics.solver_cfg.entries[0].solver_cfg.njmax = None
    cfg.sim.physics.use_cuda_graph = True
    cfg.sim.physics.solver_cfg.entries[1].solver_cfg.max_active_cell_count = n * args.envs
    with launch_simulation(cfg, dict(headless=True, device="cuda:0", visualizer=[])):
        env = ManagerBasedEnv(cfg)
        try:
            env.reset(seed=42)
            pose = torch.as_tensor(initial["body_pose"], device=env.device).repeat(args.envs, 1)
            pose[:, :3] += env.scene.env_origins
            env.scene["body5"].write_root_pose_to_sim_index(root_pose=pose)
            env.scene["body5"].write_root_velocity_to_sim_index(
                root_velocity=torch.as_tensor(initial["body_velocity"], device=env.device).repeat(args.envs, 1)
            )
            NewtonManager.forward()
            model = NewtonManager.get_model()
            assert model.particle_count == n * args.envs
            assert model.body_count == args.envs
            command = torch.zeros(args.envs, 6, device=env.device)

            def operation(k):
                for _ in range(4):
                    env.step(command)

            result = measure(operation, args.steps, args.warmup)
            state = NewtonManager.get_state_0()
            finite = np.isfinite(state.particle_q.numpy()).all() and np.isfinite(state.body_q.numpy()).all()
            return dict(
                particles_per_environment=n,
                timings=result,
                finite=bool(finite),
                physics_dt_s=cfg.sim.dt,
                substeps_per_sample=4,
                mpm_max_iterations=25,
                grid_spacing_m=0.04,
                outer_cuda_graph=NewtonManager._graph is not None,
                inner_cuda_graph=NewtonManager._solver._entries["sand"].solver._use_cuda_graph,
                memory=memory(),
                scope="Four coupled MPM + MJWarp env.step calls, including manager dispatch",
                limitation=(
                    "Visible x/v matched to learned initialization; MPM internal stress/deformation starts reset"
                ),
            )
        finally:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["network", "learned", "mpm"], required=True)
    parser.add_argument("--envs", type=int, required=True)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--compile", action="store_true", help="Fuse the neural network before CUDA graph capture")
    parser.add_argument(
        "--checkpoint", type=Path, default=ROOT / "particle_nerd_plan/results/shovel/training/model/best.pt"
    )
    parser.add_argument(
        "--episode", type=Path, default=ROOT / "experiments/particle_gnn/shovel/reference/wide-circle.hdf5"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.compile and args.backend == "mpm":
        parser.error("--compile applies only to the network and learned backends")
    if min(args.envs, args.steps, args.warmup) < 1:
        parser.error("Counts must be positive")
    torch.set_num_threads(1)
    wp.init()
    wp.config.log_level = wp.LOG_WARNING
    torch.cuda.reset_peak_memory_stats()
    result = dict(
        backend=args.backend,
        environments=args.envs,
        simulated_interval_s=0.02,
        steps=args.steps,
        warmup=args.warmup,
        gpu=torch.cuda.get_device_name(),
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        motion="Gravity and initial recorded body velocity; zero external wrench",
        memory_before=memory(),
    )
    with torch.inference_mode():
        result.update(mpm(args) if args.backend == "mpm" else learned(args))
    result.setdefault("memory", memory())
    result["environment_steps_per_second"] = args.envs * 1000 / result["timings"]["wall"]["median_ms"]
    result["per_environment_ms"] = result["timings"]["wall"]["median_ms"] / args.envs
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("RESULT", json.dumps({k: v for k, v in result.items() if k not in ("neural_config", "timings")}), flush=True)
    if not result["finite"]:
        raise RuntimeError("Finite/convergence checks failed; timings are not a validated rollout")


if __name__ == "__main__":
    main()
