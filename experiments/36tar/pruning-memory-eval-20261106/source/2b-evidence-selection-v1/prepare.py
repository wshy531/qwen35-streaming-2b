"""Select by source/category/hash only; never load existing model scores."""
from pathlib import Path
import hashlib,json,re
from collections import Counter,defaultdict

ROOT=Path(__file__).resolve().parents[2]
PREP=ROOT/'runs/evidence_selection_prepare_20261029_01'
BASE=ROOT/'datasets/S-EMBER/897886cc967b2d0e6e45f1200605f5d3b0ffbb23'
TRANS=ROOT/'runs/perception_transition_prepare_20261027_01/inputs'
PARTNER=ROOT/'runs/memory_collab_prepare_20261026_01/partner_readonly'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def order(s):return hashlib.sha256(('evidence-selection-20261029:'+s).encode()).hexdigest()

def main():
    if PREP.exists():raise FileExistsError('retain frozen panel')
    paths=[BASE/'sember_mcq.jsonl',PARTNER/'requests-no-labels.json',TRANS/'tasks_LOCAL_ONLY.json',
           TRANS/'cases_LOCAL_ONLY.json',ROOT/'runs/2b-perception-transition-219310/completed.json']
    mcq={r['question_id'].lower():r for r in map(json.loads,paths[0].read_text(encoding='utf-8').splitlines())}
    prior=read(paths[1]);trans=read(paths[2]);names={r['source_name'] for r in trans}
    official_names={r['source_path'].split('/')[-1] for r in prior}
    allowed=names|set(sorted(official_names-names,key=order)[:16-len(names)])
    buckets=defaultdict(list)
    for row in prior:
        if row['source_path'].split('/')[-1] not in allowed:continue
        m=mcq[row['id'].split(':')[-1].lower()]
        assert m['question']==row['question'] and m['question_time']==row['query_time_s']
        buckets[m['question_category']].append((row,m))
    for v in buckets.values():v.sort(key=lambda pair:order(pair[0]['id']))
    selected=[]
    while len(selected)<46:
        added=False
        for k in sorted(buckets):
            if buckets[k] and len(selected)<46:selected.append(buckets[k].pop(0));added=True
        if not added:raise ValueError('not enough source-independent fixed candidates')
    tasks=[];labels={};bindings={}
    for row,m in selected:
        t=dict(id=row['id'],cohort='official_opened_reference',question=row['question'],options=row['options'],
            source_name=row['source_path'].split('/')[-1],query_time_s=row['query_time_s'],
            category=m['question_category'],role='official_reference')
        tasks.append(t);labels[t['id']]=m['correct_index']
        a,b=float(m['answer_start_time']),float(m['answer_end_time'])
        valid=0<=a<=b<=t['query_time_s']
        bindings[t['id']]=dict(intervals=[[a,b]] if valid else [],raw_interval=[a,b],
            causal_interval_valid=valid,label_scope='official_interval_proxy_not_independent_evidence_review',
            exact_evidence_pair=False,readability_unknown=True)
    for row in trans:
        t={k:row[k] for k in ('id','question','options','source_name','query_time_s','role','case_id','stage')}
        t.update(cohort='assistant_reviewed_transition_development',category='state_transition')
        tasks.append(t);labels[t['id']]=row['correct_index']
        pts=[r['pts_s'] for r in row['evidence_frames']] if 'evidence_frames' in row else row['evidence_interval_s']
        bindings[t['id']]=dict(intervals=[[min(pts),max(pts)]],causal_interval_valid=True,
            exact_evidence_pair=True,label_scope='previous_single_assistant_review',readability_unknown=True)
    assert len(tasks)==64 and len({t['id'] for t in tasks})==64
    # Keep labels AND evidence span bindings out of a future learned selector's requests.
    save(PREP/'inputs/requests.json',tasks);save(PREP/'audit/labels_LOCAL_ONLY.json',labels)
    save(PREP/'audit/evidence_bindings_LOCAL_ONLY.json',bindings)
    profiles=read(paths[-1])['profiles'];byprofile={(p['source_name'],p['query_time_s']):p for p in profiles}
    from selection import choose
    coverage=[]
    for t in tasks:
        p=byprofile.get((t['source_name'],t['query_time_s']))
        if t['cohort']!='assistant_reviewed_transition_development':continue
        if p is None:raise ValueError('existing true snapshot missing')
        nodes=p['bank']['nodes'];hist,recent=nodes[:-4],nodes[-4:]
        result=choose(hist,recent,bindings[t['id']]['intervals'],t['query_time_s'])
        coverage.append(dict(task_id=t['id'],**result,history_nodes=hist,recent_nodes=recent))
    save(PREP/'audit/recorded_node_coverage.json',coverage)
    protocol=dict(status='panel_frozen_CPU_only',tasks=64,sources=len({t['source_name'] for t in tasks}),
        categories=dict(Counter(t['category'] for t in tasks)),roles=dict(Counter(t['role'] for t in tasks)),
        observed_true_snapshots=len(coverage),official_node_snapshots_pending=46,
        arms=['recent','full','lexical','time_matched_nodes','nonmatching_nodes'],shifts=list(range(5)),
        max_answers=1600,max_new_tokens=8,max_generation_tokens=12800,
        independent_test=False,selected_by_model_results=False,trained=False,GPU_jobs=0,
        time_matched_is_diagnostic_not_automatic=True,official_intervals_not_gold_node_support=True,
        missing_control='explicitly unavailable; do not relabel a related node as irrelevant',
        missing_evidence='keep original task and score; mark condition unavailable; no raw-frame supplementation',
        future_GPU_budget_draft=dict(GPUs=1,CPUs=4,RAM_GiB=24,minutes=45,comment='gx_llm',partition='g078t'),
        runtime_media_audit_pending=True,execution_preflight_pending=True)
    save(PREP/'protocol.json',protocol)
    save(PREP/'source_lock.json',{p.relative_to(ROOT).as_posix():sha(p) for p in paths})
    print(json.dumps(protocol,ensure_ascii=False))
if __name__=='__main__':main()
