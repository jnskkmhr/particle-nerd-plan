# G1 reference simulation and recording

This script runs the current native IsaacLab G1 + MPM–MJWarp task. It uses the
cached pretrained policy, native robot actuation and flat sand settings. The robot
starts at the center of the sand instead of walking across the approach platform.
The isolated experiment module is loaded with the main checkout's current runtime
and task implementation; neither existing checkout is modified.

## Set up the command

```bash
cd ~/isaac/IsaacLab
export PYTHONPATH="$HOME/isaac/particle_nerd/worktrees/IsaacLab-scalability/examples${PYTHONPATH:+:$PYTHONPATH}"
```

The default checkpoint is
`source/isaaclab_tasks/isaaclab_tasks/contrib/g1_29dof_mpm/benchmark/data/checkpoints/5f1xyc21/model_39998.pt`
in this checkout. Override it with `--checkpoint`; its adjacent
`wandb_config.json` must describe the policy architecture and observation groups.

## Watch policy walking and turning in Newton RTX

```bash
uv run --no-sync python -m particle_nerd.data_collection.g1_reference \
  --duration 5 --mode policy --command 0.15 0 0 \
  --command-end 0.1 0.05 0.25 --switch-time 2.5
```

Commands are robot-frame forward velocity [m/s], lateral velocity [m/s], and yaw
rate [rad/s]. The policy drives actuators; robot motion is dynamically simulated.
The window closes at the requested duration; increase `--duration` to watch longer.

## Record policy trajectories in parallel

```bash
uv run --no-sync python -m particle_nerd.data_collection.g1_reference \
  --headless --num-envs 2 --duration 5 --mode policy \
  --command 0.15 0 0 --command-end 0.1 0.05 0.25 --switch-time 2.5 \
  --output /tmp/g1-policy-reference.hdf5
```

Each environment has its own randomized initial state. This pilot applies the
same command schedule to the batch; vary `--seed`, commands and switch time across
invocations. Two environments have been tested; 100 environments are not yet
validated for this 118,440-particle G1 task. The recorder stops the entire batch
at its first native termination and marks reset transitions invalid.

## Record a reference video

```bash
uv run --no-sync python -m particle_nerd.data_collection.g1_reference \
  --headless --duration 5 --mode policy --command 0.15 0 0 \
  --command-end 0.1 0.05 0.25 --video-dir /tmp/g1-reference-video
```

Use a fresh video directory to preserve previous clips. Video and HDF5 output can
be enabled together. Recording/rendering are excluded from the scaling benchmark.

## Run policy standing

```bash
uv run --no-sync python -m particle_nerd.data_collection.g1_reference \
  --mode policy --command 0 0 0 --duration 5
```

This supplies a zero velocity command to the policy, which still produces joint
commands to balance the robot.

## Add sustained, changing joint-action perturbations

```bash
uv run --no-sync python -m particle_nerd.data_collection.g1_reference \
  --mode policy-noise --command 0.15 0 0 --noise-amplitude 0.1 --duration 5
```

A bounded random joint-action target changes every 0.2 s and is smoothed between
changes. These are action perturbations, not external force pushes on the torso.

## Run zero joint action

```bash
uv run --no-sync python -m particle_nerd.data_collection.g1_reference \
  --mode zero --duration 5
```

Zero normalized joint-position action holds the task's nominal targets through
its actuators. It does not mean zero actuator torque or policy standing.

## Run random joint actions

```bash
uv run --no-sync python -m particle_nerd.data_collection.g1_reference \
  --mode random --noise-amplitude 0.1 --duration 5
```

These use the same bounded, smoothed action process without the policy. Falls and
short episodes are retained with termination metadata; they are not counted as
successful five-second locomotion trajectories.

## Change sand internal friction

```bash
uv run --no-sync python -m particle_nerd.data_collection.g1_reference \
  --mode policy --command 0.15 0 0 --material-friction 0.4 --duration 5
```

This overrides only sand internal friction. Sand–robot contact friction stays at
the native task setting.

## Recorded state and storage

The current flat task contains 118,440 particles in a 6 × 2 × 0.3 m sand volume,
with 0.04 m MPM grid spacing. Physics runs at 200 Hz; records are written at 50 Hz.
Particle spacing and rendering radius are different from the MPM grid spacing.

Each file contains the initial frame plus the resulting states, actions,
commands, transition validity, particle masses/radii, body poses/velocities,
joint states, native collision metadata and material/configuration metadata.
Particle/body positions are relative to their environment origin; vector axes
remain world-aligned. Body quaternions use xyzw ordering.
`action[k]` produces state frame `k+1`. A five-second run has 251 frames and
250 transitions. Native foot material wrenches are last-substep samples (5 ms),
not forces averaged over the 20 ms recorded transition. They must not be treated
as integrated force targets without adding substep accumulation.

This is a G1 reference recorder and preview, not yet a plug-in replacement for
the existing single-intruder training dataset loader. Robot geometry encoding
and the articulated learned runner remain separate work.

The completed five-second policy pilot occupies 714,830,317 bytes with LZF.
Thus 5,000 trajectories need approximately 3.57 TB at this particle count.
Keep the FP32 particle velocities for now; positions alone halve the particle
payload but lose explicit velocity labels. Initial settling is included in the
pilot; the production dataset should also sample five-second windows after
settling so settling does not dominate all trajectories.

## Proposed dataset size and coverage

Start with a 200-trajectory coverage pilot (about 143 GB). Check successful
standing, touchdown/liftoff, forward/lateral walking, turning, speed changes,
reversals and disturbed recovery before expanding. A first full budget is
5,000 trajectories of five seconds: 4,000 training, 500 validation and 500 test.
Split by complete trajectory seed and motion schedule, never adjacent frames.
Use 500/1,000/2,000/4,000-training-trajectory learning curves with fixed evaluation
sets to determine whether more data helps. Five thousand is a budget, not an
established sample requirement.

Suggested coverage: 70% policy locomotion with varied commands, 15% perturbed
policy actions, 10% policy standing and 5% zero/random-action diagnostics. Count
terminated episodes separately. Reserve unseen friction values and command
combinations for additional generalization tests. Flat G1 foot contacts do not
replace separate shovel scooping, transport and sand-separation data.
