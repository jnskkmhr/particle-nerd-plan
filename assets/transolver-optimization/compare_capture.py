"""Alternate Torch/Warp capture owners for exactly the same compiled network."""
from pathlib import Path
import json
import torch
import warp as wp
from particle_nerd.data_collection.dataset import initialize_predictor
from particle_nerd.benchmarks.scalability import make_inputs, capture_network, measure

root = Path('/home/jkamohara/isaac/particle_nerd')
torch.set_num_threads(1)
p, initial = initialize_predictor(
    root/'particle_nerd_plan/results/shovel/training/model/best.pt',
    root/'experiments/particle_gnn/shovel/reference/wide-circle.hdf5',
    compile_model=True,
)
values = make_inputs(p, torch.as_tensor(initial['body_pose'],device='cuda'), torch.as_tensor(initial['body_velocity'],device='cuda'))
batch = tuple(x.unsqueeze(0).repeat(32,*([1]*x.ndim)).contiguous() for x in values)
with torch.inference_mode():
    torch_graph, torch_output = capture_network(p.model,batch)
    current = torch.cuda.current_stream()
    stream = torch.cuda.Stream()
    stream.wait_stream(current)
    warp_stream = wp.stream_from_torch(stream)
    launch_stream = wp.stream_from_torch(current)
    with torch.cuda.stream(stream), wp.ScopedStream(warp_stream), torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
        for _ in range(3): p.model(*batch)
        with wp.ScopedCapture() as capture:
            warp_output = p.model(*batch).contiguous()
    current.wait_stream(stream)
    operations = {'torch': lambda k: torch_graph.replay(), 'warp': lambda k: wp.capture_launch(capture.graph,stream=launch_stream)}
    rows = []
    for repeat in range(4):
        for owner in (['torch','warp'] if repeat % 2 == 0 else ['warp','torch']):
            timings = measure(operations[owner],60,20)
            rows.append(dict(repeat=repeat,capture_owner=owner,timings=timings))
            print(owner,timings['wall']['median_ms'],flush=True)
    result = dict(environments=32,particles_per_environment=len(p.x),compiled=True,
        max_output_difference_m=(torch_output-warp_output).abs().max().item(), measurements=rows)
    (root/'experiments/transolver-compile/capture-comparison.json').write_text(json.dumps(result,indent=2)+'\n')
