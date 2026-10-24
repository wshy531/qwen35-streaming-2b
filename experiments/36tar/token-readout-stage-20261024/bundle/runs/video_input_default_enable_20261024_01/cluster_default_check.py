"""Four CPU-only canonical adapter/default/cancellation checks."""
import json,os,sys
from pathlib import Path
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/video-input-cpu-v1'))
from integration import long_source,short_packets,DEFAULT_DECODER_CONFIG
from source import Packet
assert DEFAULT_DECODER_CONFIG=='auto4'
groups=json.loads((ROOT/'runs/2b_auto4_cost_prepare_20261024_01/inputs/groups.json').read_text())
env={'FramePacket':Packet};checks=[]
for kind in ('long','short'):
    group=next(g for g in groups if (g['cohort']=='stream34')==(kind=='long'))
    key='cluster_media' if (ROOT/'repo').exists() else 'local_media'
    path=ROOT/group[key];assert path.resolve().is_relative_to(ROOT.resolve())
    for config in (None,'slice1'):
        kwargs={} if config is None else dict(decoder_config=config)
        if kind=='long':
            src=long_source(env,path,group,**kwargs);it=src.frames()
            first,second=next(it),next(it);packets=src.take_pair([first.frame_id,second.frame_id])
            assert src.config==('auto4' if config is None else config)
            it.close();src.close();assert not src.buffer
        else:
            it=short_packets(env,path,group,**kwargs);packets=[next(it),next(it)];it.close()
        assert [p.frame_id for p in packets]==[r['frame_id'] for r in group['frames'][:2]]
        assert [p.timestamp for p in packets]==[r['pts_s'] for r in group['frames'][:2]]
        checks.append(dict(kind=kind,config='default_auto4' if config is None else config,frames=[p.frame_id for p in packets],cancelled=True))
assert 'torch' not in sys.modules
receipt=dict(status='passed',checks=checks,GPU_jobs=0,Torch_imported=False)
with (Path(__file__).parent/'cluster_default_check.json').open('x') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt))
