"""Independent accounting of frozen inputs, callbacks and captured bank tensors."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent
sys.path.insert(0,str(ROOT/'experiments/2b-memory-readability-v1'))
from prepare import read,sha


def main():
    folder=OUT/'gpu_217784';result=folder/'results';inputs=folder/'input'
    completed=read(result/'completed.json');protocol=read(inputs/'protocol.json')
    assert completed['status']=='complete' and completed['pretrained_loads']==1
    assert completed['answers']==36 and completed['visual_encodings']==1350 and completed['generated_tokens']<=288
    assert completed['protocol_sha256']==sha(inputs/'protocol.json')==sha(OUT/'runtime_inputs/protocol.json')
    for name,h in protocol['inputs_sha256'].items():assert sha(inputs/name)==h,name
    for name,h in read(inputs/'runtime_lock.json').items():
        p=ROOT/name if not name.startswith('repo/') else ROOT/'publish/qwen35-streaming-2b'/name[5:]
        assert sha(p)==h,name
    for name,h in read(OUT/'analysis_source_lock.json').items():assert sha(ROOT/name)==h,name
    auth=read(inputs/'GPU_authorization.json');assert auth['approved'] and auth['protocol_sha256']==sha(inputs/'protocol.json')
    assert read(inputs/'cpu_cluster.json')['status']=='passed'
    rows=read(inputs/'requests.json');by_id={r['id']:r for r in rows}
    arms=set(['A_raw_old','B_compressed_old','C_raw_old_recent','D_compressed_old_recent','E_full_pairwise','F_bank_reproduction'])
    predictions={}
    for line in (result/'predictions.jsonl').read_text(encoding='utf-8').splitlines():
        p=json.loads(line);key=(p['id'],p['arm']);assert key not in predictions and p['status']=='ok'
        assert p['id'] in by_id and p['arm'] in arms and p['input_tokens']+8<=32768
        assert p['visual_tokens']<=16384 and len(p['generated_ids'])<=8
        predictions[key]=p
    assert len(predictions)==36 and set(predictions)=={(r['id'],a) for r in rows for a in arms}
    callbacks={k:[] for k in predictions};times={k:[] for k in predictions}
    for line in (result/'tokens.jsonl').read_text(encoding='utf-8').splitlines():
        t=json.loads(line);key=(t['id'],t['arm']);assert t['token_index']==len(callbacks[key])
        callbacks[key].append(t['token_id']);times[key].append(t['monotonic_s'])
    assert sum(map(len,callbacks.values()))==completed['generated_tokens']
    assert all(callbacks[k]==p['generated_ids'] and times[k]==sorted(times[k]) for k,p in predictions.items())
    snapshots=read(result/'snapshots.json');assert len(snapshots)==6
    total_bytes=0;stored=0
    for snap in snapshots:
        path=result/snap['path'];assert sha(path)==snap['sha256'];total_bytes+=path.stat().st_size
        row=by_id[snap['id']];assert snap['nodes']==row['expected_bank_nodes']
        with np.load(path,allow_pickle=False) as bank:
            assert len(bank.files)==3*len(snap['nodes'])
            for i,node in enumerate(snap['nodes']):
                f,m,c=[bank[f'n{i}_{key}'] for key in ('features','mass','coords')]
                n=node['tokens'];assert f.shape==(n,2048) and m.shape==(n,1) and c.shape==(n,3)
                assert f.dtype==np.float32 and m.dtype==np.float32 and np.issubdtype(c.dtype,np.integer)
                assert np.isfinite(f).all() and np.isfinite(m).all() and (m>0).all()
                assert abs(float(m.sum())-48*node['original_pairs'])<1e-3
                assert sorted(set(c[:,0].tolist()))==node['surviving_source_pair_ids']
                assert (c[:,0]>=node['first']).all() and (c[:,0]<=node['last']).all()
                stored+=sum(a.nbytes for a in (f,m,c))
    assert total_bytes<24*2**20
    manifest=read(folder/'manifest.json');assert not manifest['queue'].strip()
    assert '217784|COMPLETED|0:0|' in manifest['sacct']
    report=dict(status='passed',all36_unique_answers=True,callbacks_match_all_generated_IDs=True,
        raw_bank_snapshots=6,snapshots_compressed_bytes=total_bytes,snapshot_tensor_bytes=stored,
        snapshot_arrays_finite_mass_conserved_and_source_coords_valid=True,
        input_protocol_authorization_CPU_and91_runtime_locks_valid=True,
        analysis_source_and_labels_unchanged=True,queue_empty=True,
        raw_feature_archive_peak_bytes=max(s['temporary_raw_feature_bytes'] for s in snapshots),
        diagnostic_archive_NOT_candidate_memory=True,
        completed=completed,sacct=manifest['sacct'])
    with (OUT/'independent_final_audit.json').open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
    print(report)


if __name__=='__main__':main()
