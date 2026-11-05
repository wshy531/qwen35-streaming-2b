"""Small real-explanation pilot, only after separate <=2-CNY paid approval."""
import json,os,stat,sys,time,urllib.request,urllib.error,hashlib
from real_pilot_api import R,P,read,payload,reserve
OUT=R/'runs/evidence-real-judge-20261102-01'
def main():
    assert R.resolve()==__import__('pathlib').Path('/home/u2025010462/share/czy/swt/my_pape/qwen35-streaming-2b-36tar').resolve()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
    auth=read(P/'real_API_authorization.json');assert auth['approved'] and auth['cap_cny']==2
    assert hashlib.sha256((P/'real_pilot_lock.json').read_bytes()).hexdigest()==auth['lock_sha256']
    for n,h in read(P/'real_pilot_lock.json').items():assert hashlib.sha256((R/n).read_bytes()).hexdigest()==h,n
    secret=R/'.secrets/deepseek.json';assert not secret.is_symlink() and not secret.parent.is_symlink() and secret.stat().st_uid==os.getuid() and stat.S_IMODE(secret.stat().st_mode)==0o600
    key=read(secret)['DEEPSEEK_API_KEY'];assert isinstance(key,str) and key and not any(c.isspace() for c in key)
    sys.path.insert(0,str(R/'scripts'));from deepseek_api_setup import NoRedirect
    opener=urllib.request.build_opener(NoRedirect());OUT.mkdir(exist_ok=False);start=time.monotonic();spent=0.;attempts=uploaded=0;pending=None
    def save(n,v):(OUT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    def ledger():save('budget.json',dict(cap_cny=2,peak_upper_cny=spent,requests=attempts,uploaded_bytes=uploaded,pending=pending))
    def call(case,stage,facts=None):
        nonlocal spent,attempts,uploaded,pending
        assert time.monotonic()-start<1200 and attempts<36
        p=payload(case,stage,facts);res=reserve(p);upper=res['cny_upper']+256*8e-6;raw=json.dumps(p,ensure_ascii=False).encode()
        assert spent+upper<=2 and uploaded+len(raw)<=200*2**20
        attempts+=1;uploaded+=len(raw);pending=dict(opaque_id=case['opaque_id'],stage=stage,cny=upper);ledger()
        req=urllib.request.Request('https://api.deepseek.com/chat/completions',data=raw,
            headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'},method='POST')
        with opener.open(req,timeout=90) as response:body=response.read(512*1024+1)
        assert len(body)<=512*1024
        reply=json.loads(body);usage=reply['usage'];pt=usage['prompt_tokens'];ct=usage['completion_tokens']
        assert type(pt)is int and type(ct)is int and min(pt,ct)>=0
        spent+=pt*2e-6+ct*8e-6;pending=None;ledger()
        choice=reply['choices'][0];text=(choice['message'].get('content') or '').replace(key,'[REDACTED]')
        with (OUT/'responses.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(dict(opaque_id=case['opaque_id'],stage=stage,content=text,usage=usage,model=reply.get('model'),finish_reason=choice.get('finish_reason')),ensure_ascii=False)+'\n')
        assert pt<=res['input_token_upper'] and ct<=768 and spent<=2
        try:return json.loads(text)
        except (ValueError,TypeError):return dict(raw_unparsed=text,processing_status='needs_review_not_content_error')
    results=[]
    try:
        for case in read(P/'real_pilot_LOCAL_ONLY.json'):
            facts=call(case,'fact')
            support=call(case,'support',facts) if isinstance(facts.get('claims'),list) else dict(processing_status='fact_extraction_needs_review_not_content_error')
            results.append(dict(opaque_id=case['opaque_id'],fact=facts,support=support))
            save('judgments.json',results);print(json.dumps(dict(explanations=len(results),requests=attempts,peak_upper_cny=spent)),flush=True)
        save('completed.json',dict(status='complete_real_explanation_pilot',explanations=len(results),requests=attempts,peak_upper_cny=spent,
            seconds=time.monotonic()-start,GPU=0,training=0,automatic_retry=False,not_gold_accuracy=True))
    except Exception as e:
        ledger();save('failed.json',dict(type=type(e).__name__,http_status=e.code if isinstance(e,urllib.error.HTTPError) else None,
            explanations=len(results),requests=attempts,peak_upper_cny=spent,pending=pending,automatic_retry=False));raise SystemExit(2)
if __name__=='__main__':main()
