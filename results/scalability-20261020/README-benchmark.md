# Environment-count benchmark

The benchmark measures the current shovel checkpoint at 16,384 particles per
world. It is not a G1 benchmark or an equal-accuracy comparison against MPM.
Each sample advances 20 ms: one learned step or four 5 ms coupled MPM steps.
Initialization, rendering, recording and compilation are excluded. Reported wall
time includes Python dispatch and waits for GPU completion.

## Set up the benchmark command

```bash
cd ~/isaac/particle_nerd/worktrees/IsaacLab-scalability
export PARTICLE_NERD_PYTHON="$HOME/isaac/IsaacLab-particle-nerd/.venv/bin/python"
RUN=examples/particle_nerd/run.sh
```

## Profile all three implementations

```bash
for environments in 1 2 4 8 16 32; do
  for backend in network learned mpm; do
    "$RUN" -m particle_nerd.benchmarks.scalability \
      --backend "$backend" --envs "$environments" --warmup 20 --steps 60 \
      --output "/tmp/particle-nerd-scaling/${backend}-${environments}.json"
  done
done
```

Run these sequentially with no other GPU experiment. `network` measures a
captured batched Transolver forward on fixed inputs. `learned` rebuilds features,
batches the forward pass, updates particles, estimates contact forces and advances
MJWarp. The feature, force and rigid stages still dispatch per environment.
`mpm` uses the native batched MPM–MJWarp collector with 25 MPM solver iterations.

Attention never crosses environment boundaries. The checkpoint weights are
unchanged. Neural inference uses BF16 autocast. Neural, force and rigid operations
have separate CUDA graphs; eager feature generation remains outside capture.
The initial visible particle/body state is shared across implementations, but
MPM stress history is reset. No external dragging force is applied in this test.

`stages` comes from a separate instrumented continuation, not the primary timing
samples. `device_used_GiB` is a device-wide allocator snapshot, including other
GPU users, and is not a process peak-memory measurement. The script overwrites
its JSON output; choose a new output directory to retain earlier measurements.
