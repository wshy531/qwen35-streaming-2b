"""Fixed diagnostic cells; no answer-driven dropping or additional runs."""
import hashlib,json,sys
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[2];RUN=ROOT/'runs/2b_early9_diagnostic_20261020_01'
from score_memory_eval import score,paired
ARMS=['old_full','old_merged','old_full_recent4','old_merged_recent4','recent4','text_only','full_replay']
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    rows=read(ROOT/'runs/2b_natural_eval_20261020_01/scorer_rows_local.json');ids={r['id'] for r in read(RUN/'requests.json')};rows=[r for r in rows if r['id'] in ids]
    if len(rows)!=9:raise ValueError('all9 diagnostic rows')
    if sys.argv[1]=='freeze':
        save(RUN/'analysis_lock.json',dict(source_sha256=sha(__file__),scorer_rows_sha256=sha(ROOT/'runs/2b_natural_eval_20261020_01/scorer_rows_local.json'),comparisons=['old_full -> old_merged','old_full -> old_full_recent4','old_merged -> old_merged_recent4','recent4 -> old_full_recent4','recent4 -> old_merged_recent4'],all9_including_base_failures=True,no_bootstrap_claim_for_6_related_sources=True,scope='case diagnosis, no global capability superiority, manual old windows are not automatic retrieval'))
        print('diagnostic analysis frozen');return
    lock=read(RUN/'analysis_lock.json')
    if sha(__file__)!=lock['source_sha256'] or sha(ROOT/'runs/2b_natural_eval_20261020_01/scorer_rows_local.json')!=lock['scorer_rows_sha256']:raise ValueError('analysis identity drift')
    job=read(RUN/'submission_result.json')['job_id'];base=RUN/('gpu_'+job)/'result'
    completed=read(base/'completed.json');ps=[json.loads(s) for s in (base/'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    if len(ps)!=63 or len({(p['id'],p['arm']) for p in ps})!=63 or completed['records']!=63 or completed['protocol_sha256']!=sha(RUN/'protocol.json'):raise ValueError('complete fixed63 required')
    scores={a:score(rows,[p for p in ps if p['arm']==a]) for a in ARMS};comparisons={}
    for a,b in [('old_full','old_merged'),('old_full','old_full_recent4'),('old_merged','old_merged_recent4'),('recent4','old_full_recent4'),('recent4','old_merged_recent4')]:comparisons[a+' -> '+b]=paired(scores[a],scores[b])
    pp={(p['id'],p['arm']):p for p in ps};ds={a:{d['id']:d for d in s['details']} for a,s in scores.items()}
    details=[dict(id=r['id'],question=r['question'],official_answer=r['options'][r['answer_index']],cells={a:dict(text=pp[r['id'],a]['text'],correct=ds[a][r['id']]['correct']) for a in ARMS}) for r in rows]
    previous=[json.loads(s) for s in (ROOT/'runs/2b_natural_eval_20261020_01/gpu_217141/result/memory/predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    old={(p['id'],p['arm']):p for p in previous};replay={}
    for a,pa in [('full_replay','full_pairwise_reference'),('text_only','text_only')]:replay[a]=sum(pp[r['id'],a]['generated_ids']==old[r['id'],pa]['generated_ids'] for r in rows)
    out=dict(status='complete',questions=9,sources=6,correct={a:s['overall']['correct'] for a,s in scores.items()},formats={a:s['overall']['statuses'] for a,s in scores.items()},comparisons=comparisons,replay_generated_IDs_equal=replay,details=details,execution=completed,limitations=['9 related diagnostic questions on6 already-open sources, not an accuracy benchmark.','Old window manually specified from full-prefix review, not automatic retrieval.','Compression changes feature values, surviving positions, labels and Token count together.','Recent-only can answer by priors or persistent cues; correctness alone does not prove history reliance.','All preselected cases retained including originally unanswerable inputs; no window/prompt revision.'])
    save(RUN/'analysis.json',out);print(json.dumps({k:v for k,v in out.items() if k not in ['details','limitations']}));print(json.dumps(details,ensure_ascii=False))
if __name__=='__main__':main()
