"""One-shot paid worker. Requires a separate exact authorization; no retries."""
import hashlib,json,os,stat,sys,time,urllib.request,urllib.error
from pathlib import Path
from payloads import R,P,read,payload,reserve,normalize,decode_response
OUT=R/'runs/evidence-demo-judge-20261102-01'
def calibration_gate(rows):
    expected={c['opaque_id']:c['LOCAL_ONLY'] for c in read(P/'calibrations_LOCAL_ONLY.json')}
    sufficient=[];contradicted=[];unknown=[];irrelevant=[]
    for r in rows:
        x=r['judgment'];e=expected[r['opaque_id']]
        if e['expected_support']=='sufficient':sufficient.append(x.get('support')=='sufficient' and x.get('core_error') is False)
        elif e['expected_fact']=='contradicted':contradicted.append(x.get('core_error') is True)
        elif e['expected_fact']=='unverifiable':unknown.append(x.get('support')!='sufficient' and (x.get('core_unverified') is True or x.get('input_coverage')=='unverifiable'))
        else:irrelevant.append(x.get('support')!='sufficient')
    assert list(map(len,(sufficient,contradicted,unknown,irrelevant)))==[8,7,4,1]
    return dict(sufficient=[sum(sufficient),8],contradicted=[sum(contradicted),7],unknown=[sum(unknown),4],irrelevant=[sum(irrelevant),1],
        passed=sum(sufficient)>=7 and sum(contradicted)>=6 and sum(unknown)>=3 and all(irrelevant) and all(not r['judgment'].get('processing_issues') for r in rows))
def main():
    assert R.resolve()==Path('/home/u2025010462/share/czy/swt/my_pape/qwen35-streaming-2b-36tar').resolve()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
    auth=read(P/'authorization.json');assert auth['approved'] and auth['cap_cny']==10 and auth['max_requests']==122
    assert hashlib.sha256((P/'execution_lock.json').read_bytes()).hexdigest()==auth['lock_sha256']
    for path,digest in read(P/'execution_lock.json').items():assert hashlib.sha256((R/path).read_bytes()).hexdigest()==digest,path
    secret=R/'.secrets/deepseek.json';assert not secret.is_symlink() and not secret.parent.is_symlink()
    assert secret.stat().st_uid==os.getuid() and stat.S_IMODE(secret.stat().st_mode)==0o600
    key=read(secret)['DEEPSEEK_API_KEY'];assert isinstance(key,str) and key and not any(c.isspace() for c in key)
    sys.path.insert(0,str(R/'scripts'));from deepseek_api_setup import NoRedirect
    opener=urllib.request.build_opener(NoRedirect());OUT.mkdir(exist_ok=False)
    start=time.monotonic();spent=0.;count=uploaded=0;pending=None;results=[];contracts={}
    def save(n,v):(OUT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    def ledger():save('budget.json',dict(cap_cny=10,peak_upper_cny=spent,requests=count,uploaded_bytes=uploaded,pending=pending))
    def call(item,stage,contract=None):
        nonlocal count,uploaded,spent,pending
        assert time.monotonic()-start<1800 and count<122
        p=payload(item,stage,contract);r=reserve(p);raw=json.dumps(p,ensure_ascii=False).encode()
        assert spent+r['cny_upper']<=10 and uploaded+len(raw)<=1024**3
        count+=1;uploaded+=len(raw);ident=item.get('opaque_id',item['task_id'])
        pending=dict(id=ident,stage=stage,reservation_cny=r['cny_upper']);ledger()
        req=urllib.request.Request('https://api.deepseek.com/chat/completions',data=raw,
            headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'},method='POST')
        with opener.open(req,timeout=90) as resp:body=resp.read(512*1024+1)
        assert len(body)<=512*1024
        reply=json.loads(body);usage=reply['usage'];pt=usage['prompt_tokens'];ct=usage['completion_tokens']
        assert type(pt)is int and type(ct)is int and min(pt,ct)>=0
        spent+=pt*2e-6+ct*8e-6;pending=None;ledger()
        choice=reply['choices'][0];text=(choice['message'].get('content') or '').replace(key,'[REDACTED]')
        with (OUT/'responses.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(id=ident,stage=stage,content=text,usage=usage,model=reply.get('model'),finish_reason=choice.get('finish_reason')),ensure_ascii=False)+'\n')
        assert pt<=r['input_token_upper'] and ct<=1024 and spent<=10
        out=normalize(decode_response(text),item,stage)
        print(json.dumps(dict(requests=count,cny=spent,id=ident)),flush=True)
        return out
    try:
        calibration=[]
        for item in read(P/'calibrations_LOCAL_ONLY.json'):
            calibration.append(dict(opaque_id=item['opaque_id'],judgment=call(item,'judge')))
            save('calibration.json',calibration)
        gate=calibration_gate(calibration);save('calibration_gate.json',gate)
        if not gate['passed']:
            save('stopped.json',dict(reason='calibration_not_sufficient_for_method_comparison',requests=count,cny=spent));return
        for item in read(P/'tasks.json'):
            contracts[item['task_id']]=call(item,'contract');save('contracts.json',contracts)
        for item in read(P/'cases_LOCAL_ONLY.json')+read(P/'repeats_LOCAL_ONLY.json'):
            contract=contracts[item['task_id']]
            results.append(dict(opaque_id=item['opaque_id'],task_id=item['task_id'],repeat_of=item.get('repeat_of'),judgment=call(item,'judge',contract)))
            save('judgments.json',results)
        save('completed.json',dict(status='completed',questions=24,explanations=72,repeats=6,requests=count,
            peak_upper_cny=spent,seconds=time.monotonic()-start,GPU=0,training=0,automatic_retry=False,not_gold_accuracy=True))
    except Exception as e:
        ledger();save('failed.json',dict(type=type(e).__name__,http_status=e.code if isinstance(e,urllib.error.HTTPError) else None,
            requests=count,peak_upper_cny=spent,pending=pending,completed_explanations=len(results),automatic_retry=False));raise SystemExit(2)
if __name__=='__main__':main()
