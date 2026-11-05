"""Paid calibration only with separately recorded explicit 10-CNY authorization."""
import json,os,stat,sys,time,urllib.request,urllib.error
from calibration_api import R,P,RUN,read,views,payload,reserve
OLD=R/'runs/evidence-judge-calibration-20261102-01'
OUT=R/'runs/evidence-judge-calibration-resume-20261102-01'
def main():
    assert R.resolve()==__import__('pathlib').Path('/home/u2025010462/share/czy/swt/my_pape/qwen35-streaming-2b-36tar').resolve()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
    authorization=read(P/'API_authorization.json');assert authorization['approved'] and authorization['cap_cny']==10
    import hashlib
    assert hashlib.sha256((P/'calibration_lock.json').read_bytes()).hexdigest()==authorization['lock_sha256']
    for name,digest in read(P/'calibration_lock.json').items():
        assert hashlib.sha256((R/name).read_bytes()).hexdigest()==digest,name
    for name,digest in read(P/'calibration_resume_lock.json').items():
        assert hashlib.sha256((R/name).read_bytes()).hexdigest()==digest,name
    secret=R/'.secrets/deepseek.json'
    assert not secret.is_symlink() and not secret.parent.is_symlink() and secret.stat().st_uid==os.getuid() and stat.S_IMODE(secret.stat().st_mode)==0o600
    key=read(secret)['DEEPSEEK_API_KEY'];assert isinstance(key,str) and key and not any(c.isspace() for c in key)
    sys.path.insert(0,str(R/'scripts'));from deepseek_api_setup import NoRedirect
    opener=urllib.request.build_opener(NoRedirect());OUT.mkdir(exist_ok=False)
    receipt=read(OLD/'budget.json');assert receipt['attempts']==1 and receipt['pending_reservation'] is None
    cached=json.loads((OLD/'responses.jsonl').read_text().splitlines()[0])
    assert cached['request_id']=='view-00' and cached['stage']=='fact'
    started=time.monotonic();spent=receipt['charged_peak_upper_cny'];attempts=1;uploaded=receipt['uploaded_bytes'];pending=None;scores=[]
    def save(name,value):(OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    def ledger():save('budget.json',dict(cap_cny=10,charged_peak_upper_cny=spent,pending_reservation=pending,attempts=attempts,uploaded_bytes=uploaded))
    def call(v,stage,fact=None):
        nonlocal attempts,spent,uploaded,pending
        if v['request_id']=='view-00' and stage=='fact':return json.loads(cached['content'])
        assert time.monotonic()-started<1200 and attempts<80
        p=payload(v,stage,fact);reservation=reserve(p);raw=json.dumps(p,ensure_ascii=False).encode()
        if spent+reservation['cny_upper']>10:raise RuntimeError('budget_bound')
        assert uploaded+len(raw)<=400*2**20
        attempts+=1;uploaded+=len(raw);pending=dict(request_id=v['request_id'],stage=stage,cny=reservation['cny_upper']);ledger()
        req=urllib.request.Request('https://api.deepseek.com/chat/completions',data=raw,
            headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'},method='POST')
        with opener.open(req,timeout=min(90,max(1,int(1200-(time.monotonic()-started))))) as response:data=response.read(256*1024+1)
        assert len(data)<=256*1024
        reply=json.loads(data);usage=reply['usage'];pt=usage['prompt_tokens'];ct=usage['completion_tokens']
        assert type(pt)is int and type(ct)is int and min(pt,ct)>=0
        spent+=pt*2e-6+ct*8e-6;pending=None;ledger()
        choice=reply['choices'][0];text=(choice['message'].get('content') or '').replace(key,'[REDACTED]')
        with (OUT/'responses.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(dict(request_id=v['request_id'],stage=stage,content=text,usage=usage,
                model=reply.get('model'),finish_reason=choice.get('finish_reason'),images=reservation['images']),ensure_ascii=False)+'\n')
        assert pt<=reservation['input_token_upper'] and ct<=512 and spent<=10
        assert not choice['message'].get('reasoning_content') and choice.get('finish_reason')=='stop'
        return json.loads(text)
    try:
        for v in views():
            fact=call(v,'fact');assert fact['verdict'] in ('supported','contradicted','unverifiable')
            refs=fact.get('references');assert isinstance(refs,list)
            material=next(m for m in read(RUN/'materials.json') if m['material_id']==v['material_id'])
            allowed={f['frame_id'] for f in material['frames']}
            import re
            normalized=[]
            for i in refs:
                if isinstance(i,str):
                    match=re.fullmatch(r'(?:Frame\s+)?([0-9]+)',i.strip());assert match is not None
                    i=int(match.group(1))
                assert type(i)is int and i in allowed
                normalized.append(i)
            fact=dict(fact,references=normalized)
            supported=call(v,'support',fact);assert supported['sufficiency'] in ('sufficient','partial','insufficient','unverifiable')
            row=dict(request_id=v['request_id'],calibration_id=v['calibration_id'],variant=v['variant'],
                fact=fact,support=supported,expected_fact=v['expected_fact_verdict'],expected_support=v['expected_sufficiency'])
            scores.append(row);save('scores.json',scores)
            print(json.dumps(dict(views=len(scores),requests=attempts,peak_upper_cny=spent)),flush=True)
        save('completed.json',dict(status='complete',views=len(scores),requests=attempts,peak_upper_cny=spent,seconds=time.monotonic()-started,GPU=0,automatic_retry=False))
    except Exception as e:
        ledger();save('failed.json',dict(type=type(e).__name__,http_status=e.code if isinstance(e,urllib.error.HTTPError) else None,
            views=len(scores),requests=attempts,peak_upper_cny=spent,pending=pending,automatic_retry=False))
        raise SystemExit(2)
if __name__=='__main__':main()
