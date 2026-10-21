"""Post-run descriptive audit using all six pre-output human witness sets."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent
OLD=ROOT/'runs/2b_long_stream_prepare_20261020_01'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def main():
    rows=read(OLD/'manual6_delayed_questions_LOCAL_ONLY.json')
    sources={s['source_path']:s for s in read(OLD/'stream_sources6.json')}
    checkpoints={c['id']:c for c in read(OUT/'independent_final_audit.json')['query_checkpoint_details']}
    evidence=[]
    for row in rows:
        frame_index={f['frame_id']:i//2 for i,f in enumerate(sources[row['source_path']]['frames'])}
        witnesses=sorted({frame_index[w['frame_id']] for w in row['witnesses']});arms={}
        for arm in ('M2','A06'):
            nodes=checkpoints[row['id']][arm]['closed_nodes']
            chosen=[n for n in nodes if any(n['first']<=i<=n['last'] for i in witnesses)]
            assert chosen and all(sum(n['first']<=i<=n['last'] for n in chosen)==1 for i in witnesses)
            arms[arm]=dict(selected_nodes=[dict(first=n['first'],last=n['last'],depth=n['depth'],groups=n['groups'],tokens=n['tokens']) for n in chosen],
                visual_tokens=sum(n['tokens'] for n in chosen),covered_pairs=sum(n['original_pairs'] for n in chosen))
        evidence.append(dict(id=row['id'],witness_pair_indices=witnesses,**arms))
    result=dict(all6_manual_cases_preserved=True,rows=evidence,model_calls=0,
        limitations=['descriptive source allocation, not automatic retrieval or extra counterfactual answers',
            'more source range in one node is not proof that the target detail was lost',
            'node depth, partition and merged representation change together; not pure depth causality'])
    with (OUT/'manual_witness_allocation_audit.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))


if __name__=='__main__':main()
