"""Untimed adapter smoke checks with the actual shared FramePacket type."""
import importlib.util,os,sys
from common import ROOT,OUT,read,save,sha
from integration import long_source,short_packets

assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
assert read(OUT/'completed.json')['cases']==32
file=ROOT/'publish/qwen35-streaming-2b/experiments/36tar/streaming-learning/code/qwen35-streaming-s4/streaming/window.py'
spec=importlib.util.spec_from_file_location('v0_window_contract',file)
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
env={'FramePacket':module.FramePacket};groups=read(OUT/'inputs/groups.json');checks=[]
for config in ('slice1','auto4'):
    group=next(g for g in groups if g['cohort']=='stream34')
    source=long_source(env,ROOT/group['local_media'],group,decoder_config=config)
    frames=iter(source.frames());ids=[next(frames).frame_id for _ in range(2)]
    packets=source.take_pair(ids)
    window=module.FrameWindow(max_frames=2)
    for packet in packets:
        assert isinstance(packet,module.FramePacket);window.append(packet)
    assert window.evidence()['frame_ids']==ids
    frames.close();source.close();assert not source.buffer and not source.completed
    checks.append(dict(config=config,cohort='long',actual_packet_and_window=True,cancel_checked=True))
    group=next(g for g in groups if g['cohort']=='T98')
    packets=short_packets(env,ROOT/group['local_media'],group,decoder_config=config)
    window=module.FrameWindow(max_frames=2)
    for _ in range(2):window.append(next(packets))
    assert window.evidence()['frame_ids']==[f['frame_id'] for f in group['frames'][:2]]
    packets.close()
    checks.append(dict(config=config,cohort='short',actual_packet_and_window=True,cancel_checked=True))
assert 'torch' not in sys.modules
save(OUT/'integration_check.json',dict(status='passed',checks=checks,packet_contract_SHA256=sha(file),
    adapter_SHA256=sha(ROOT/'experiments/video-input-cpu-v1/integration.py'),
    not_a_model_inference_test=True,GPU_jobs=0))
print(dict(status='passed',adapter_cases=len(checks),GPU_jobs=0))
