"""Diagnostic GPU kernel attribution; separate from uninstrumented timing."""
from pathlib import Path
import json
import torch
from particle_nerd.data_collection.dataset import initialize_predictor
from particle_nerd.benchmarks.scalability import make_inputs

root = Path('/home/jkamohara/isaac/particle_nerd')
torch.set_num_threads(1)
p, initial = initialize_predictor(root/'particle_nerd_plan/results/shovel/training/model/best.pt', root/'experiments/particle_gnn/shovel/reference/wide-circle.hdf5')
inputs = make_inputs(p, torch.as_tensor(initial['body_pose'],device='cuda'), torch.as_tensor(initial['body_velocity'],device='cuda'))
batch = tuple(x.unsqueeze(0).repeat(32,*([1]*x.ndim)).contiguous() for x in inputs)
result = {}
with torch.inference_mode(), torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
    for name in ['eager','compiled']:
        model = p.model if name == 'eager' else torch.compile(p.model,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
        for _ in range(10): model(*batch)
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
            for _ in range(5): model(*batch)
            torch.cuda.synchronize()
        kernels = {}
        for event in prof.events():
            if event.device_type != torch.autograd.DeviceType.CUDA:
                continue
            row = kernels.setdefault(event.name,dict(calls_per_forward=0,gpu_ms_per_forward=0))
            row['calls_per_forward'] += 1/5
            row['gpu_ms_per_forward'] += event.device_time_total/5000
        rows = [dict(kernel=name,**values) for name,values in kernels.items()]
        rows.sort(key=lambda row:-row['gpu_ms_per_forward'])
        result[name] = dict(kernels_per_forward=sum(row['calls_per_forward'] for row in rows),gpu_kernel_time_ms=sum(row['gpu_ms_per_forward'] for row in rows),kernels=rows)
        print(name,result[name]['kernels_per_forward'],result[name]['gpu_kernel_time_ms'],flush=True)
(root/'experiments/transolver-compile/kernel-profile.json').write_text(json.dumps(result,indent=2)+'\n')
