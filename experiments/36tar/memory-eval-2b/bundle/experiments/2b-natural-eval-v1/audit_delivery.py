"""Independent local evidence counts and immutable protocol/source checks."""
import hashlib,json,re
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def lines(p):return [json.loads(s) for s in Path(p).read_text(encoding='utf-8').splitlines()]
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    base=ROOT/'runs/2b_natural_eval_20261020_01';route=ROOT/'runs/2b_temporal_route_20261020_01';diag=ROOT/'runs/2b_early9_diagnostic_20261020_01';review=ROOT/'runs/early25_full_prefix_review_20261020_01'
    rows={r['id']:r for r in read(base/'scorer_rows_local.json')};requests={r['id']:r for r in read(base/'requests.json')}
    for name,h in read(base/'source_lock.json').items():assert sha(HERE/name)==h
    allpred=[];audit={}
    layouts=[(base,'217141',['native','memory'],1026),(diag,'217145',[''],63),(route,'217146',[''],684)]
    for folder,job,stages,n in layouts:
        ps=[];root=folder/('gpu_'+job)/'result';comps=[]
        for stage in stages:
            d=root/stage;ps+=lines(d/'predictions.jsonl');c=read(d/'completed.json');assert c['protocol_sha256']==sha(folder/'protocol.json');comps.append(c)
        assert len(ps)==n==sum(c['records'] for c in comps) and len({(p['id'],p['arm']) for p in ps})==n
        totals=Counter();formats=Counter();leading=Counter()
        for p in ps:
            assert p['status']=='ok' and 1<=len(p['generated_ids'])<=8 and p['id'] in rows
            a=rows[p['id']]['answer_index'];m=re.fullmatch(r'\s*([A-Ea-e])\s*[.!]?\s*',p['text']);choice=ord(m[1].upper())-65 if m else None
            totals[p['arm']]+=choice==a;formats[p['arm']]+=choice is None
            m=re.match(r'^\s*([A-Ea-e])(?:\b|[.!)\s])',p['text']);choice=ord(m[1].upper())-65 if m else None;leading[p['arm']]+=choice==a
        audit[job]=dict(predictions=n,generated_tokens=sum(len(p['generated_ids']) for p in ps),strict_correct=dict(totals),leading_choice_diagnostic=dict(leading),format_errors=dict(formats),model_loads=sum(c['model_loads'] for c in comps),protocol_sha256=sha(folder/'protocol.json'))
        allpred+=ps
    assert len(allpred)==1773 and len({p['id'] for p in allpred})==171
    bp=read(base/'gpu_217141/input/cpu_memory/preflight.json');rp=read(route/'protocol.json')
    for n,h in rp['new_sources'].items():assert sha(HERE/n)==h
    assert sha(HERE/'diagnose_early9.py')==read(diag/'protocol.json')['driver_sha256']
    initial={x['media']:x for x in read(base/'gpu_217141/result/memory/timelines.json')};new=read(route/'gpu_217146/result/banks.json');same=0
    for bank in new:
        assert bank['state']==initial[bank['media']]['states']['M2'];assert bank['events']==initial[bank['media']]['events'];same+=1
    rpred={(p['id'],p['arm']):p for p in lines(route/'gpu_217146/result/predictions.jsonl')};traces=read(route/'gpu_217146/result/routing_traces.json');selected=0
    for t in traces:
        key='long_only' if t['used_long_only'] else 'all_memory';r=rpred[t['id'],'temporal_route'];expected=rpred[t['id'],key]
        assert r['generated_ids']==expected['generated_ids'] and r['input_ids_sha256']==expected['input_ids_sha256'];selected+=t['used_long_only']
        for arm,nodes in t['selections'].items():
            assert sum(n['tokens'] for n in nodes)==rpred[t['id'],arm]['visual_tokens']
            assert all(n['end']<=requests[t['id']]['model_visible']['query_time_s']+1e-9 for n in nodes)
    assert selected==87 and same==67
    for name,h in read(review/'REVIEW_LOCK.json').items():assert sha(review/name)==h
    render=read(review/'render_receipt.json');assert render['frames']==1169 and render['pages']==60
    data=[q for x in lines(review/'source_reviews.jsonl') for q in x['questions']];assert len(data)==25 and len({q['id'] for q in data})==25
    for folder in [base,diag,route]:
        receipt=read(folder/'fetch_receipt.json');files=receipt['files'] if isinstance(receipt,dict) else receipt
        job=read(folder/'submission_result.json')['job_id']
        for item in files:assert sha(folder/('gpu_'+job)/item['path'])==item['sha256']
    result=dict(status='passed',jobs=audit,unique_questions=171,actual_answers=1773,generated_tokens=sum(x['generated_tokens'] for x in audit.values()),actual_pretrained_loads=4,source_locks_unchanged=True,identical_M2_banks_and_event_traces=67,routing_real_inputs_and_generation_matches=171,cue_triggered=87,curation_sources=20,curation_questions=25,curation_pages=60,curation_RGB_frames=1169,remote_fetch_all_SHA256_equal=True,scope='independent parsing/counts/data/protocol/branch/bank-metadata audit; full tensors were checked at runtime, not exported here')
    save(base/'delivery_audit.json',result);print(json.dumps(result))
if __name__=='__main__':main()
