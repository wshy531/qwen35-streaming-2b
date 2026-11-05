"""Freeze all 11 paired drafts and four evidence conditions; no model results used."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/diagnosis_readout_prepare_20261105_01'
REVIEW=ROOT/'runs/diagnosis_expansion_review_20261105_01'

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    OUT.mkdir(exist_ok=False)
    qs=read(ROOT/'datasets/streaming-evaluation-v1/diagnosis_candidate.json')
    grouped=defaultdict(list)
    for q in qs:grouped[q['source_id']].append(q)
    assert len(qs)==22 and len(grouped)==11
    packets={r['source_id']:r for r in read(REVIEW/'review_packets.json')}
    sources=[]
    for name,tasks in sorted(grouped.items()):
        assert len(tasks)==2 and {q['type'] for q in tasks}=={'current','history'}
        assert len({q['query_time_s'] for q in tasks})==1
        old=next(q for q in tasks if q['type']=='history')
        trace=read(REVIEW/name[:-4]/'trace.json')
        assert len(trace)%2==0
        pairs=[trace[i:i+2] for i in range(0,len(trace),2)]
        begin,end=old['evidence_bounds_s']
        selected=[i for i,p in enumerate(pairs) if p[0]['pts']<=end and p[1]['pts']>=begin
                  and p[1]['pts']<=end+.05]
        recent=list(range(len(pairs)-8,len(pairs)))
        assert 1<=len(selected)<=8 and set(selected).isdisjoint(recent)
        assert max(pairs[i][1]['pts'] for i in selected)<min(pairs[i][0]['pts'] for i in recent)
        assert [p['pts'] for i in recent for p in pairs[i]]==old['recent_input_pts']
        assert len(trace)==2*packets[name]['late_sampler']['complete_pairs']
        sources.append(dict(source_id=name,media_path=old['media_path'],media_sha256=old['media_sha256'],
            query_time_s=old['query_time_s'],early_pair_indices=selected,recent_pair_indices=recent,
            pairs=pairs,selected_pair_indices=sorted(set(selected+recent))))
    save(OUT/'questions.json',qs)
    save(OUT/'sources.json',sources)
    plan=dict(sources=11,questions=22,arms=['recent','early','combined','text'],
        formal_answers=88,warmup_answers=4,max_answers=92,max_new_tokens=128,
        maximum_generated_tokens=11776,causal_trace_pairs=sum(len(s['pairs']) for s in sources),
        selected_visual_pairs=sum(len(s['selected_pair_indices']) for s in sources),warmup_visual_calls=1,
        model_loads=1,precision='float32',GPU=1,CPUs=4,system_memory_GiB=24,minutes=20,
        canvas=[288,384],sampler='V0Pairs auto4 causal 1FPS',
        visual_tokens_per_pair=108,largest_combined_pairs=16,context_limit=4096,
        prompt='B',primary_option_shifts=[0],seed=20261105,
        output_limit_MiB=32,training=False,API=False,downloads=False,requeue=False,retries=0,
        decoder='seek exact previously sampled PTS/RGB; encode only the 159 evidence pairs',
        interpretation='Human-selected evidence readout diagnostic, not automatic retrieval or streaming cost.',
        labels_used_in_inference=False,all_22_questions_retained=True,
        labels_not_changed_by_model_outputs=True,
        admission='candidate stays exploratory; no candidate algorithm is being selected by these labels')
    assert plan['causal_trace_pairs']==1843 and plan['selected_visual_pairs']==159
    save(OUT/'plan.json',plan)
    keys=read(ROOT/'datasets/streaming-evaluation-v1/answer_keys_LOCAL_ONLY.json')
    save(OUT/'labels_LOCAL_ONLY.json',{q['id']:keys[q['id']] for q in qs})
    save(OUT/'input_provenance.json',{
        'datasets/streaming-evaluation-v1/diagnosis_candidate.json':sha(ROOT/'datasets/streaming-evaluation-v1/diagnosis_candidate.json'),
        'runs/diagnosis_expansion_review_20261105_01/review_packets.json':sha(REVIEW/'review_packets.json')})
    print(plan)

if __name__=='__main__':main()
