"""Freeze all prior six-case development tasks, independent of new method results."""
import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parents[2];P=R/'runs/midprune_memory_prepare_20261102_01'
OLD=R/'runs/perception_transition_prepare_20261027_01/inputs'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):
    with (P/n).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def main():
    P.mkdir(exist_ok=True)
    tasks=read(OLD/'tasks_LOCAL_ONLY.json');sources=read(OLD/'sources_LOCAL_ONLY.json')
    assert len(tasks)==18 and len(sources)==5 and len({q['case_id'] for q in tasks})==6
    assert sum(q['role']=='current' for q in tasks)==12 and sum(q['role']=='old' for q in tasks)==6
    public=[{k:q[k] for k in ('id','case_id','role','stage','source_name','query_time_s','question','options')} for q in tasks]
    labels={q['id']:{k:v for k,v in q.items() if k not in ('question','options','source_name','recent_frames')} for q in tasks}
    for s in sources:
        times={q['query_time_s'] for q in tasks if q['source_name']==s['name']}
        assert times and max(times)==s['max_query_s']
        assert len(s['sampled_trace'])%2==0
        assert times <= {x['pts_s'] for x in s['sampled_trace'][1::2]}
    pairs=sum(len(s['sampled_trace'])//2 for s in sources)
    ck='runs/hard-mask-main-220062/mask_ce_step1237.pt'
    result=read(R/'runs/prune-checkpoints-220226/analysis.json')
    assert result['development_ranking'][0]=='step1237'
    spec=dict(stage='real interface/development pilot, not generalization',questions=18,sources=5,cases=6,
        current_questions=12,history_questions=6,option_shifts=[0],views=['full','fixed12','heuristic11','mask_ce'],
        policies=['M0','M2','A06','LS'],matched_coverage_controls=['fixed12/M0_same8','heuristic11/M0_same8','mask_ce/M0_same8'],
        answers=342,max_generated_tokens=342*128,visual_calls=4*pairs,real_pairs=pairs,model_loads=1,
        checkpoint=ck,checkpoint_sha256=sha(R/ck),budget=dict(partition='g078t',comment='gx_llm',GPU=1,cpus=4,memory_GiB=24,minutes=45,output_MiB=64),
        source_access='Previously authorized five S-EMBER partner files, read only; no directory scan or write',
        media_root='/home/u2025010462/share/czy/wsy/data/S-EMBER/897886cc967b2d0e6e45f1200605f5d3b0ffbb23/videos',
        prompt='Existing answer plus 1-3 observations protocol B, original option order, max128',
        content_score='Original assistant development labels, no independent reviewer; no new API judgement',
        memory='Bounded banks only; original visual coordinates; own-path event descriptor; matched temporal coverage only for M0_same8',
        stop='Numerical/input/causal/budget/legacy-full parity error stops. No retry, training, download or API. Small paired score is descriptive, no promotion or rule tuning.')
    save('questions.json',public);save('labels_LOCAL_ONLY.json',labels);save('sources.json',sources);save('protocol.json',spec)
    save('provenance.json',{str(p.relative_to(R)).replace('\\','/'):sha(p) for p in [OLD/'tasks_LOCAL_ONLY.json',OLD/'sources_LOCAL_ONLY.json',R/'runs/prune-checkpoints-220226/analysis.json']})
    print(json.dumps(spec))
if __name__=='__main__':main()
