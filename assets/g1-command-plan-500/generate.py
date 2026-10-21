"""Create a reproducible proposed G1 command schedule and integrate its XY paths."""
from pathlib import Path
from collections import Counter
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

ROOT = Path('/home/jkamohara/isaac/particle_nerd')
OUT = ROOT / 'particle_nerd_plan/assets/g1-command-plan-500'
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(20261020)
DT = 0.02
T = np.arange(251) * DT
GROUPS = [
    ('straight', 'Straight walking', 100, '#386481'),
    ('curved', 'Curved walking', 100, '#24796d'),
    ('change', 'Speed / direction changes', 100, '#b07a34'),
    ('lateral', 'Lateral / diagonal walking', 50, '#79679c'),
    ('perturbed', 'Perturbed policy: nominal path', 75, '#ba6d6a'),
    ('standing', 'Policy standing', 50, '#6e8060'),
    ('diagnostic', 'Zero / random actions\nStarts only; no path prediction', 25, '#637079'),
]


def integrate(initial, commands):
    """Exact SE(2) integration of each constant body-frame velocity interval."""
    path = np.empty((len(commands) + 1, 3))
    path[0] = initial
    for k, (vx, vy, wz) in enumerate(commands):
        x, y, yaw = path[k]
        end_yaw = yaw + wz * DT
        if abs(wz) < 1e-10:
            c, s = np.cos(yaw) * DT, np.sin(yaw) * DT
        else:
            c = (np.sin(end_yaw) - np.sin(yaw)) / wz
            s = (np.cos(yaw) - np.cos(end_yaw)) / wz
        path[k + 1] = (x + vx*c - vy*s, y + vx*s + vy*c, end_yaw)
    return path


def draw_commands(group):
    if group == 'straight':
        a = np.array([RNG.uniform(.05, .28), 0., 0.]); b = a.copy()
    elif group == 'curved':
        a = np.array([RNG.uniform(.08, .35), RNG.uniform(-.05, .05), RNG.choice([-1,1])*RNG.uniform(.15,.6)])
        b = a.copy()
    elif group == 'change':
        a = np.array([RNG.uniform(.1,.35), RNG.uniform(-.08,.08), RNG.uniform(-.4,.4)])
        b = np.array([RNG.uniform(-.22,-.05) if RNG.random()<.5 else RNG.uniform(.05,.3), RNG.uniform(-.12,.12), RNG.uniform(-.6,.6)])
    elif group == 'lateral':
        a = np.array([RNG.uniform(0,.12), RNG.choice([-1,1])*RNG.uniform(.08,.24), RNG.uniform(-.15,.15)])
        b = a.copy()
        if RNG.random()<.5: b[1] *= -1
    elif group == 'perturbed':
        a = np.array([RNG.uniform(.05,.3), RNG.uniform(-.08,.08), RNG.uniform(-.45,.45)])
        b = np.array([RNG.uniform(.02,.25), RNG.uniform(-.1,.1), RNG.uniform(-.45,.45)])
    else:
        a = np.zeros(3); b = a.copy()
    switch_step = int(RNG.integers(75,176))
    commands = np.where((np.arange(250) < switch_step)[:,None], a, b)
    return a, b, switch_step, commands


# Verify the coordinate transform and yaw integration with analytic cases.
assert np.allclose(integrate([0,0,np.pi/2], np.tile([.2,0,0],(250,1)))[-1,:2], [0,1])
assert np.allclose(integrate([0,0,0], np.tile([1,0,1],(250,1)))[-1,:2], [np.sin(5),1-np.cos(5)])
trials=[]; paths=[]; rejected=0
for group,label,count,color in GROUPS:
    val_count = 8 if count==75 else 2 if count==25 else count//10
    splits = ['train']*(count*4//5) + ['validation']*val_count + ['test']*(count-count*4//5-val_count)
    RNG.shuffle(splits)
    for local_idx, split in enumerate(splits):
        for attempt in range(10000):
            initial = np.array([RNG.uniform(-.05,.05), RNG.uniform(-.05,.05), RNG.uniform(-np.pi,np.pi)])
            a,b,switch_step,commands = draw_commands(group)
            path=integrate(initial,commands)
            # Reject complete schedules; never clip their positions or velocities.
            if np.max(np.abs(path[:,:2])) <= 1.15: break
            rejected+=1
        else: raise RuntimeError('Could not sample an in-bounds schedule')
        mode = 'policy-noise' if group=='perturbed' else ('zero' if local_idx<12 else 'random') if group=='diagnostic' else 'policy'
        trials.append(dict(trial_id=len(trials)+1,group=group,mode=mode,split=split,
            simulation_seed=120000+len(trials),initial_x_m=float(initial[0]),initial_y_m=float(initial[1]),
            initial_yaw_rad=float(initial[2]),switch_time_s=switch_step*DT,
            command_start=a.tolist(),command_end=b.tolist(),duration_s=5.,
            action_noise_amplitude=.1 if mode in ('policy-noise','random') else 0.,
            integrated_path_interpretation='no velocity-control prediction' if group=='diagnostic' else 'ideal tracking of body-frame velocity command'))
        paths.append(path)
paths=np.asarray(paths)
assert len(trials)==500 and Counter(t['split'] for t in trials)==dict(train=400,validation=50,test=50)
assert np.isfinite(paths).all() and np.max(np.abs(paths[:,:,:2]))<=1.15
manifest=dict(status='proposed command schedule; not yet executed',seed=20261020,
    sand_dimensions_m=[3,3,.3],coordinate_origin='center of initial sand area; world XY axes',
    command_frame='robot heading frame: vx forward [m/s], vy left [m/s], wz counterclockwise yaw [rad/s]',
    policy_hz=50,proposed_physics_and_recording_hz=200,planned_base_center_inset_m=.35,
    initial_state='XY uniform +/-0.05 m; yaw uniform -pi to pi. Initial yaw randomization is proposed, unlike the earlier fixed-yaw pilot.',
    rejection_rule='Reject schedules whose integrated base center leaves +/-1.15 m in either axis; no clipping.',
    rejected_candidates=rejected,trials=trials)
(OUT/'commands.json').write_text(json.dumps(manifest,indent=2)+'\n')
fields=['trial_id','group','mode','split','simulation_seed','initial_x_m','initial_y_m','initial_yaw_rad','switch_time_s','vx_start_mps','vy_start_mps','wz_start_radps','vx_end_mps','vy_end_mps','wz_end_radps','duration_s','action_noise_amplitude']
with (OUT/'commands.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=fields,lineterminator="\n");writer.writeheader()
    for tr in trials:
        row={k:tr[k] for k in fields if k in tr}
        row.update(zip(fields[9:15],tr['command_start']+tr['command_end']))
        writer.writerow(row)
np.savez_compressed(OUT/'integrated_paths.npz',time_s=T,xy_yaw=paths)
# Compact paths for interactive trial inspection; full precision is in NPZ.
(OUT/'paths.json').write_text(json.dumps(np.round(paths,6).tolist(),separators=(',',':'))+'\n')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,4,figsize=(15,8.5),layout='constrained')

def setup(ax,title):
    ax.add_patch(Rectangle((-1.5,-1.5),3,3,facecolor='#f0e9dc',edgecolor='#947b59',lw=1.3,zorder=0))
    ax.add_patch(Rectangle((-1.15,-1.15),2.3,2.3,fill=False,edgecolor='#83928c',ls='--',lw=1,zorder=1))
    ax.set(xlim=(-1.6,1.6),ylim=(-1.6,1.6),aspect='equal',xlabel='World x from sand center [m]',ylabel='World y from sand center [m]',title=title)
    ax.title.set_fontsize(10)
    ax.set_xticks([-1.5,0,1.5]);ax.set_yticks([-1.5,0,1.5]);ax.grid(alpha=.18)

setup(axes.flat[0],'All 500 trials\n425 moving + 75 zero-command trials')
for j,(group,label,count,color) in enumerate(GROUPS):
    ax=axes.flat[j+1];setup(ax,f'{label}\n{count} trials')
    for i,tr in enumerate(trials):
        if tr['group']!=group:continue
        for target in [axes.flat[0],ax]:
            if group in ('standing','diagnostic'):
                target.scatter(*paths[i,0,:2],s=12,color=color,marker='x' if group=='diagnostic' else 'o',alpha=.5)
            else:
                target.plot(paths[i,:,0],paths[i,:,1],color=color,alpha=.24 if target==axes.flat[0] else .4,lw=.8)
                target.scatter(*paths[i,-1,:2],s=5,color=color,alpha=.45)
fig.suptitle('Planned G1 base paths from integrated velocity commands — 500 trials × 5 s',fontsize=16)
fig.supxlabel('Shaded square: 3 × 3 m sand area. Dashed square: proposed base-center planning margin. Dots on moving paths: endpoints.\nUniform initial yaw; body-frame commands rotated into world XY. No physics, tracking error, or disturbance response is simulated.',fontsize=10)
fig.savefig(OUT/'trajectory-plan.png',dpi=180)
fig.savefig(OUT/'trajectory-plan.svg')
plt.close(fig)
fig,ax=plt.subplots(figsize=(8,8),layout='constrained')
setup(ax,'500 planned trials · 5 seconds each')
for group,label,count,color in GROUPS:
    for i,tr in enumerate(trials):
        if tr['group']!=group:continue
        if group in ('standing','diagnostic'):
            ax.scatter(*paths[i,0,:2],s=10,color=color,marker='x' if group=='diagnostic' else 'o',alpha=.6)
        else:
            ax.plot(paths[i,:,0],paths[i,:,1],color=color,alpha=.38,lw=.8)
            ax.scatter(*paths[i,-1,:2],s=7,color=color,alpha=.5)
handles=[Line2D([0],[0],color=color,lw=2,label=f'{label.split(chr(10))[0]} ({count})') for _,label,count,color in GROUPS]
ax.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,-.12),ncol=2,fontsize=9,frameon=False)
fig.suptitle('G1 base XY paths from velocity-command integration',fontsize=14)
fig.savefig(OUT/'overview.png',dpi=180)
fig.savefig(OUT/'overview.svg')
plt.close(fig)
for p in OUT.glob('*.svg'):p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')
summary=dict(trials=len(trials),groups=dict(Counter(t['group'] for t in trials)),splits=dict(Counter(t['split'] for t in trials)),max_abs_xy_m=float(np.max(np.abs(paths[:,:,:2]))),rejected_candidates=rejected)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
