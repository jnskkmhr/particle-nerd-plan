"""Compare the same trained network with and without compilation; both use CUDA graphs."""
from pathlib import Path
import argparse
import json
import time
import h5py
import numpy as np
import torch
import warp as wp
from particle_nerd.data_collection.dataset import initialize_predictor
from particle_nerd.benchmarks.scalability import make_inputs, capture_network, measure

parser=argparse.ArgumentParser();parser.add_argument('--envs',type=int,required=True);parser.add_argument('--variant',choices=['eager','compile','autotune','precision'],required=True)
parser.add_argument('--capture',choices=['torch','warp'],default='torch')
args=parser.parse_args();torch.set_num_threads(1)
root=Path('/home/jkamohara/isaac/particle_nerd')
checkpoint=root/'particle_nerd_plan/results/shovel/training/model/best.pt'
episode=root/'experiments/particle_gnn/shovel/reference/wide-circle.hdf5'
p,state=initialize_predictor(checkpoint,episode)
inputs=make_inputs(p,torch.as_tensor(state['body_pose'],device='cuda'),torch.as_tensor(state['body_velocity'],device='cuda'))
batch=tuple(x.unsqueeze(0).repeat(args.envs,*([1]*x.ndim)).contiguous() for x in inputs)
with torch.inference_mode():
 network=p.model if args.variant=='eager' else torch.compile(p.model,fullgraph=True,dynamic=False,**({'mode':'max-autotune-no-cudagraphs'} if args.variant=='autotune' else {'options':{'triton.cudagraphs':False, 'emulate_precision_casts':args.variant=='precision'}}))
 started=time.perf_counter()
 if args.capture=='torch':
  graph,output=capture_network(network,batch)
  replay=graph.replay
 else:
  stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
  warp_stream=wp.stream_from_torch(stream)
  launch_stream=wp.stream_from_torch(torch.cuda.current_stream())
  with torch.cuda.stream(stream),wp.ScopedStream(warp_stream),torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
   for _ in range(3):network(*batch)
   with wp.ScopedCapture() as capture:
    output=network(*batch).contiguous()
  torch.cuda.current_stream().wait_stream(stream)
  replay=lambda:wp.capture_launch(capture.graph,stream=launch_stream)
 torch.cuda.synchronize();setup=time.perf_counter()-started
 timings=measure(lambda k:replay(),60,20)
 checks=[]
 with h5py.File(episode) as f:
  for frame in [2,20,70,120,180,250,300]:
   p.history.copy_(torch.as_tensor(np.stack([f['particle_position'][j] for j in range(frame-2,frame+1)],axis=1),device='cuda'))
   step=int(f['frame_steps'][frame]);values=make_inputs(p,torch.as_tensor(f['body_pose'][step],device='cuda'),torch.as_tensor(f['body_velocity'][step],device='cuda'))
   for dst,value in zip(batch,values,strict=True):dst.copy_(value.unsqueeze(0).expand_as(dst))
   replay();torch.cuda.synchronize();actual=output.clone()
   with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):reference=p.model(*batch)
   difference=actual-reference
   checks.append(dict(frame=frame,max_abs_difference_m=difference.abs().max().item(),rms_difference_m=difference.square().mean().sqrt().item(),reference_rms_m=reference.square().mean().sqrt().item(),finite=bool(torch.isfinite(actual).all())))
result=dict(variant=args.variant,environments=args.envs,particles_per_environment=len(p.x),cuda_graph=True,capture_owner=args.capture,torch_version=torch.__version__,setup_seconds=setup,timings=timings,checks=checks)
out=root/f'experiments/transolver-compile/{args.variant}-{args.envs}{'-warp' if args.capture=='warp' else ''}.json';out.write_text(json.dumps(result,indent=2)+'\n')
print('RESULT',json.dumps({k:v for k,v in result.items() if k not in ['timings','checks']}),flush=True);print('MEDIAN_MS',timings['wall']['median_ms']);print('MAX_DIFFERENCE_M',max(c['max_abs_difference_m'] for c in checks))
