"""Audit already captured real nodes, never load a pretrained model."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'runs/2b_memory_readability_prepare_20261020_01'
OUT=ROOT/'runs/2b_closed_balance_prepare_20261020_01'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def main():
    requests=read(OLD/'requests6_LOCAL.json');snapshots={s['id']:s for s in read(OLD/'gpu_217784/results/snapshots.json')}
    closed={};last={}
    events=ROOT/'runs/2b_long_stream_prepare_20261020_01/gpu_217776/memory/events.jsonl'
    for line in events.open(encoding='utf-8'):
        e=json.loads(line)
        if e['arm']!='M2':continue
        if e['event']['kind']=='pair':last[e['source_path']]=e['state']['bank']['details']['closed_tokens']//48
        elif e['event']['kind']=='query':closed[e['event']['query_id']]=last[e['source_path']]
    rows=[]
    for request in requests:
        snap=snapshots[request['id']];path=OLD/'gpu_217784/results'/snap['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==snap['sha256']
        with np.load(path,allow_pickle=False) as bank:
            for i in request['selected_node_indices']:
                node=snap['nodes'][i];mass=bank[f'n{i}_mass'];coords=bank[f'n{i}_coords']
                assert mass.shape==(48,1) and abs(float(mass.sum())-48*node['original_pairs'])<1e-3
                rows.append(dict(id=request['id'],node_index=i,is_closed_root=i<closed[request['id']],
                    first=node['first'],last=node['last'],pairs=node['original_pairs'],depth=node['depth'],groups=node['groups'],
                    mass_sum=float(mass.sum()),max_token_mass=float(mass.max()),
                    max_token_mass_fraction=float(mass.max()/mass.sum()),
                    surviving_destination_pair_count=len(set(coords[:,0].tolist())),
                    coordinate_min_max=[int(coords[:,0].min()),int(coords[:,0].max())],
                    snapshot_sha256=snap['sha256']))
    assert len(rows)==7 and all(r['is_closed_root'] for r in rows)
    result=dict(status='audited_saved_real_tensors',nodes=rows,GPU_jobs=0,pretrained_loads=0,
        limits=['mass is original-token contribution count, NOT attention or semantic importance',
            'concentrated mass also occurs in correct cases; not sufficient cause',
            'surviving source positions are anchors, not complete support of averaged content',
            'depth and mass are descriptive; no new model answer or causal isolation'])
    with (OUT/'actual_selected_node_audit.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))


if __name__=='__main__':main()
