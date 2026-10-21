"""Build the measured compiler report; no benchmark is run by this script."""
from pathlib import Path
import hashlib
import json
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path('/home/jkamohara/isaac/particle_nerd')
source = root/'experiments/transolver-compile'
site = root/'particle_nerd_plan'
assets = site/'assets/transolver-optimization'
assets.mkdir(exist_ok=True)
def read(name):
    return json.loads((source/name).read_text())
def wall(record):
    return record['timings']['wall']['median_ms']
counts = [1,8,32]
neural = {str(n): {v: read(f'{v}-{n}.json') for v in ['eager','compile']} for n in counts}
full = {str(n): {v: read(f'full-{v}-{n}.json') for v in ['eager','compiled']} for n in counts}
rollouts = read('rollout-agreement.json')
repeat = read('repeat/rollout-agreement.json')
precision = read('precision/rollout-agreement.json')
# The original validation harness used the comparison label "compiled" for
# its second run. Label the two diagnostic comparisons accurately in the report.
for case in repeat.values():
    case['repeat'] = case.pop('compiled')
for case in precision.values():
    case['precision_preserving_compiled'] = case.pop('compiled')
capture = read('capture-comparison.json')
kernels = read('kernel-profile.json')
metadata = dict(
    gpu='RTX 5090', particles_per_environment=16384, precision='BF16 autocast',
    checkpoint_sha256=hashlib.sha256((site/'results/shovel/training/model/best.pt').read_bytes()).hexdigest(),
    neural=neural, full_simulation=full, graph_owner_comparison=capture,
    coupled_rollout_agreement=rollouts, uncompiled_repeatability=repeat,
    precision_preserving_compilation=precision, kernel_profile=kernels,
    source_hashes={},
)
worktree = root/'worktrees/IsaacLab-scalability'
for filename in ['data_collection/dataset.py','benchmarks/scalability.py','transolver/model.py','transolver/inference.py','runner/simulation.py']:
    metadata['source_hashes'][filename] = hashlib.sha256((worktree/'examples/particle_nerd'/filename).read_bytes()).hexdigest()
(assets/'measurements.json').write_text(json.dumps(metadata,indent=2)+'\n')
for filename in ['benchmark.py','compare_capture.py','validate_rollouts.py','profile_kernels.py','publish_report.py']:
    shutil.copyfile(source/filename,assets/filename)
shutil.copyfile(worktree/'examples/particle_nerd/benchmarks/scalability.py',assets/'scalability.py')

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(11.5,4),layout='constrained')
for axis,data,new,title in [(axes[0],neural,'compile','Neural forward only'),(axes[1],full,'compiled','Complete learned simulation')]:
    xs=np.arange(3)
    for offset,variant,label,color in [(-.18,'eager','Uncompiled + CUDA graph','#426d86'),(.18,new,'Compiled + CUDA graph','#248173')]:
        values=[wall(data[str(n)][variant]) for n in counts]
        bars=axis.bar(xs+offset,values,.34,label=label,color=color)
        axis.bar_label(bars,fmt='%.1f',padding=3,fontsize=10)
    axis.set_xticks(xs,[str(n) for n in counts]);axis.set_xlabel('Parallel environments')
    axis.set_ylabel('Whole-batch wall time [ms]');axis.set_title(title);axis.set_ylim(0,axis.get_ylim()[1]*1.15)
axes[0].legend(fontsize=9,loc='upper left')
fig.savefig(assets/'latency.svg');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(11.5,3.8),layout='constrained')
for path,color in [('wide-circle','#426d86'),('figure-eight','#248173')]:
    a=np.load(source/f'{path}-eager-rollout.npz')['body_pose']
    b=np.load(source/f'{path}-compiled-rollout.npz')['body_pose']
    center=np.linalg.norm(a[:,:3]-b[:,:3],axis=1)*1000
    qa,qb=a[:,3:],b[:,3:]
    dot=np.abs(np.sum(qa*qb,axis=1)/(np.linalg.norm(qa,axis=1)*np.linalg.norm(qb,axis=1)))
    angle=np.rad2deg(2*np.arccos(np.clip(dot,0,1)))
    t=.02*np.arange(1,len(a)+1)
    axes[0].plot(t,center,color=color,label=path)
    axes[1].plot(t,angle,color=color,label=path)
for axis in axes:axis.set_xlabel('Simulated time [s]');axis.legend();axis.grid(alpha=.18)
axes[0].set_ylabel('Shovel center difference [mm]');axes[1].set_ylabel('Shovel orientation difference [degrees]')
fig.savefig(assets/'rollout-difference.svg');plt.close(fig)
for path in assets.glob('*.svg'):
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')

table_rows=''
for n in counts:
    a,b=wall(neural[str(n)]['eager']),wall(neural[str(n)]['compile'])
    c,d=wall(full[str(n)]['eager']),wall(full[str(n)]['compiled'])
    table_rows+=f'<tr><td>{n}</td><td>{a:.2f} → {b:.2f}</td><td>{a/b:.2f}×</td><td>{c:.2f} → {d:.2f}</td><td>{c/d:.2f}×</td></tr>'
capture_medians={owner:float(np.median([row['timings']['wall']['median_ms'] for row in capture['measurements'] if row['capture_owner']==owner])) for owner in ['torch','warp']}
agreement_rows=''
for path,data in rollouts.items():
    a=data['agreement']
    agreement_rows+=f"<tr><td>{path}</td><td>{1000*a['body_center_rms_distance_m']:.1f} / {1000*a['body_center_max_distance_m']:.1f} mm</td><td>{a['body_orientation_max_degrees']:.1f}°</td><td>{1000*a['particle_position_rms_distance_m']:.2f} mm</td><td>{a['force_rms_component_N']:.1f} N</td></tr>"
html=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Transolver compilation and CUDA graphs — Particle NeRD</title>
<style>body{{max-width:1120px;margin:35px auto;padding:0 22px;font:18px/1.6 system-ui;color:#24383d;background:#f7f9f7}}h1{{font-size:38px;line-height:1.2}}h2{{font-size:27px;margin-top:42px}}h3{{font-size:21px}}a{{color:#386481}}nav{{font-size:15px;border-bottom:1px solid #cad8d4;padding-bottom:16px}}table{{width:100%;border-collapse:collapse;font-size:16px}}th,td{{padding:10px;text-align:left;border-bottom:1px solid #cbd7d8}}th{{background:#eaf0ee}}pre{{padding:18px;background:#eaf0ee;white-space:pre-wrap;overflow-wrap:anywhere;font-size:14px}}img{{width:100%;height:auto}}figure{{margin:25px 0}}figcaption,.small{{font-size:15px;color:#566970}}.note{{padding:18px;border-left:4px solid #24796d;background:#eaf0ee}}.caution{{border-color:#a56f32;background:#f3eee5}}.scroll{{overflow-x:auto}}code{{overflow-wrap:anywhere}}</style></head><body>
<nav><a href="index.html">Methodology</a> · <a href="scalability-g1.html">Batch scaling and G1 dataset</a> · <a href="compute-profile.html">Compute profile</a></nav>
<h1>Optimizing the trained Transolver</h1><p>20 October 2026 · RTX 5090 · unchanged shovel checkpoint · 16,384 particles per environment</p>
<p class="note"><strong>Measured result:</strong> compilation improves neural latency by 1.54–1.81× and complete simulation latency by 1.17–1.23×. Both implementations already use CUDA graphs. Warp and Torch graph capture give essentially the same runtime for the same compiled kernels.</p>
<p class="note caution"><strong>Accuracy limitation:</strong> compiled BF16 arithmetic produces small one-step differences that grow during coupled simulation. The wide-circle shovel trajectory diverges substantially. Compilation remains an optional experiment; it is not enabled by default or established as an accuracy-preserving replacement.</p>
<h2>1. What changes?</h2><table><thead><tr><th>Operation</th><th>Current implementation</th><th>Experiment</th></tr></thead><tbody>
<tr><td>Particle/collider features</td><td>Warp GPU kernels, per environment</td><td>Unchanged</td></tr>
<tr><td>Particle NeRD forward</td><td>Torch operations, BF16, CUDA graph</td><td>Compile the same model with TorchInductor, then capture its forward</td></tr>
<tr><td>Particle update, analytical contact, rigid dynamics</td><td>Warp/MJWarp; separate contact and rigid graphs</td><td>Unchanged</td></tr>
<tr><td>Checkpoint and inputs</td><td>Same trained weights, particle history and collider features</td><td>No retraining or architecture change</td></tr></tbody></table>
<p>The reported copies are <strong>GPU device-side copies and dtype conversions</strong>. They are not particle transfers to the CPU. The compiler combines operations such as normalization, conversion, addition and division into fewer GPU kernels. Matrix multiplications remain necessary.</p>
<p>In a separate five-forward GPU profile at 32 environments, kernel launches decrease from <strong>{round(kernels['eager']['kernels_per_forward'])} to {round(kernels['compiled']['kernels_per_forward'])}</strong> per forward. Summed kernel time falls from {kernels['eager']['gpu_kernel_time_ms']:.1f} to {kernels['compiled']['gpu_kernel_time_ms']:.1f} ms. Generated kernels explicitly fuse softmax/conversion/division and LayerNorm/addition/conversion. These are instrumented diagnostic measurements, separate from the latency table below.</p>
<p>Standard FlashAttention is already used for attention between 64 latent slices. The expensive particle-to-slice operations and particle MLPs remain outside that attention kernel. At 32 environments, one FP32 slice-weight tensor occupies 512 MiB: 32 environments × 4 heads × 16,384 particles × 64 slices × 4 bytes. Compilation reduces intermediate memory traffic, but does not remove this scaling cost.</p>
<h2>2. Measured latency</h2><figure><img src="assets/transolver-optimization/latency.svg" alt="Uncompiled and compiled neural and complete simulation latency at one, eight and 32 environments"><figcaption>Whole-batch synchronized wall time. Every environment advances 20 ms per complete simulation step. Both neural implementations use CUDA graphs.</figcaption></figure>
<div class="scroll"><table><thead><tr><th>Environments</th><th>Neural: before → after [ms]</th><th>Speedup</th><th>Complete: before → after [ms]</th><th>Speedup</th></tr></thead><tbody>{table_rows}</tbody></table></div>
<p>32-environment compiled simulation still spends about 7.5 ms on feature construction and batch copies, 30.3 ms on the neural forward, and 45.9 ms on particle updates, force estimation and rigid dynamics in a separate instrumented continuation. Feature and contact work still dispatch separately for each environment.</p>
<p><strong>Conclusion:</strong> compilation helps, but does not solve parallel-environment scalability. The earlier MPM measurement was 64.0 ms at 32 environments; the compiled learned simulation is still slower. That historical MPM measurement is context, not a new simultaneous or equal-accuracy comparison.</p>
<h3>Does Warp graph capture improve on Torch graph capture?</h3><p>Capturing the same compiled forward through each framework gives <strong>{capture_medians['torch']:.2f} ms with Torch and {capture_medians['warp']:.2f} ms with Warp</strong> at 32 environments, with identical outputs. These are medians of four 60-sample runs per framework, with alternating measurement order. The difference is within observed run-to-run variation.</p>
<p>A CUDA graph replays existing CUDA work with less CPU dispatch. It does not automatically fuse kernels or reduce matrix multiplication. TorchInductor changes the kernels; graph capture changes their launch. A mixed Torch/Warp graph is possible using a shared CUDA stream and zero-copy tensor views. Rewriting the network in Warp would only help if its kernels execute the computation more efficiently.</p>
<h2>3. Numerical agreement over five seconds</h2><p>On seven recorded input frames, compiled and uncompiled displacement predictions differ by at most 0.234 mm in any coordinate across the tested batch sizes. One-step RMS component differences are at most 0.0183 mm. These small differences do <strong>not</strong> establish rollout agreement.</p>
<figure><img src="assets/transolver-optimization/rollout-difference.svg" alt="Shovel center and orientation differences grow during five-second compiled and uncompiled coupled rollouts"><figcaption>Each rollout uses its own predicted particles and dynamic shovel state. The same moving drag target applies forces to each run; the shovel pose is not prescribed. Curves compare compiled with uncompiled simulation, not either method against MPM.</figcaption></figure>
<div class="scroll"><table><thead><tr><th>Drag target</th><th>Shovel center RMS / max difference</th><th>Max orientation difference</th><th>Particle RMS distance</th><th>Material-force component RMS difference</th></tr></thead><tbody>{agreement_rows}</tbody></table></div>
<p>Particle RMS distances pool every tenth recorded step and all particles. Shovel pose and force differences use all 250 steps. Forces refer to the estimated material force on the shovel, in world coordinates. The two runs' force RMS scales are approximately 181 N for wide-circle and 222 N for figure-eight. Some particle outliers are much larger than the pooled RMS: maximum particle differences are 0.56 m and 0.087 m respectively.</p>
<p>Repeating the uncompiled simulation reproduced particle positions, shovel positions and force arrays exactly in these tests. Enabling <code>emulate_precision_casts=True</code> did not resolve compiled rollout divergence: maximum shovel orientation differences remained 76.2° and 5.0°. All runs stayed finite and passed contact-solver convergence checks; these checks do not establish physical accuracy.</p>
<p class="note"><strong>Decision:</strong> keep <code>--compile</code> optional. Before making it the default, evaluate compiled and uncompiled models against the same MPM validation trajectories, including force and long-horizon motion errors. Independently, batch feature generation and contact/rigid operations to address the remaining per-environment cost.</p>
<h2>4. Run the experiment</h2><pre>cd ~/isaac/particle_nerd/worktrees/IsaacLab-scalability
export PARTICLE_NERD_PYTHON="$HOME/isaac/IsaacLab-particle-nerd/.venv/bin/python"
RUN=examples/particle_nerd/run.sh

"$RUN" -m particle_nerd.benchmarks.scalability --backend learned --envs 32 \\
  --output /tmp/learned-uncompiled-32.json
"$RUN" -m particle_nerd.benchmarks.scalability --backend learned --envs 32 --compile \\
  --output /tmp/learned-compiled-32.json</pre>
<p>Use <code>--backend network</code> for neural-only timing. The interactive <code>particle_nerd.runner.play</code> command also accepts <code>--compile</code>; the example README includes a complete command. TorchInductor's internal graph capture is disabled because the predictor already owns the outer neural graph.</p>
<h3>Measurement conditions</h3><ul><li>Torch 2.11.0+cu130; RTX 5090; four layers, width 128, four heads, 64 latent slices. Same checkpoint SHA-256 recorded in the raw data.</li><li>20 warm-up steps, 60 timed steps for each latency measurement. Compilation, capture, recording and rendering are excluded. Fresh paired measurements are used; values vary from the earlier scaling table.</li><li>Batch timing uses the settled shovel scene with gravity and initial velocity, without dragging. Five-second validation uses wide-circle and figure-eight drag forces, one environment, 50 Hz particles and 200 Hz rigid dynamics.</li><li>Full simulation has no MPM solve. The analytical solver uses effective material mass scale 180, relaxation 0.075 and a 2,000-iteration limit; five-second runs needed at most 10 solver iterations.</li><li>Five existing model tests pass, including environment isolation; the compiled headless runner completes ten steps. Changed-file formatting/lint checks pass. The repository-wide changelog gate still fails on unrelated existing fragments.</li></ul>
<p><a href="assets/transolver-optimization/measurements.json">Raw measurements and source hashes</a> · <a href="assets/transolver-optimization/scalability.py">Full simulation benchmark</a> · <a href="assets/transolver-optimization/benchmark.py">Neural benchmark</a> · <a href="assets/transolver-optimization/compare_capture.py">Torch/Warp comparison</a> · <a href="assets/transolver-optimization/validate_rollouts.py">Coupled validation</a> · <a href="assets/transolver-optimization/profile_kernels.py">GPU kernel profile</a></p>
<h3>Implementation references</h3><p><a href="https://pytorch.org/blog/why-is-pytorch-compile-so-fast-kernel-fusion/">PyTorch kernel fusion</a> · <a href="https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/">CUDA graph replay</a> · <a href="https://nvidia.github.io/warp/v1.15/user_guide/interoperability_pytorch.html">Warp/Torch shared streams, tensor views and mixed capture</a></p>
</body></html>'''
(site/'transolver-optimization.html').write_text(html+'\n')
print('Report written:',site/'transolver-optimization.html')
