"""All six fixed manual witnesses; descriptive provenance, no extra answers."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def main():
    old=ROOT/'runs/2b_long_stream_prepare_20261020_01'
    rows=read(old/'manual6_delayed_questions_LOCAL_ONLY.json')
    sources={s['source_path']:s for s in read(old/'stream_sources6.json')}
    checkpoints={c['id']:c for c in read(OUT/'independent_final_audit.json')['query_checkpoint_details']}
    answers={(d['id'],d['arm']):d for d in read(OUT/'strict_results.json')['details']};records=[]
    for row in rows:
        index={f['frame_id']:i//2 for i,f in enumerate(sources[row['source_path']]['frames'])}
        witnesses=sorted({index[w['frame_id']] for w in row['witnesses']});arms={};cp=checkpoints[row['id']]
        for arm,b in cp['banks'].items():
            selected=[n for n in b['nodes'] if any(n['first']<=i<=n['last'] for i in witnesses)]
            covered=[i for i in witnesses if any(n['first']<=i<=n['last'] for n in selected)]
            mass=cp['mass_profiles'][arm]
            chosen=[dict(first=n['first'],last=n['last'],depth=n['depth'],groups=n['groups'],tokens=n['tokens'],
                exact_pair=n['exact_pair'],**{k:next(p for p in mass if (p['first'],p['last'])==(n['first'],n['last']))[k]
                    for k in ('max_token_mass','max_mass_fraction')}) for n in selected]
            arms[arm]=dict(selected_nodes=chosen,covered_witness_pairs=covered,missing_witness_pairs=sorted(set(witnesses)-set(covered)),
                selected_tokens=sum(n['tokens'] for n in selected),covered_source_pairs=sum(n['original_pairs'] for n in selected),
                answer=answers[row['id'],arm])
        records.append(dict(id=row['id'],witness_pair_indices=witnesses,arms=arms))
    result=dict(rows=records,all6_preserved=True,model_calls=0,
        limitations=['manual provenance lookup not automatic retrieval','mass is source contribution count, not attention/information loss',
            'more separate roots or less mass concentration not sufficient proof of readable event order'])
    with (OUT/'manual_witness_allocation_audit.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
