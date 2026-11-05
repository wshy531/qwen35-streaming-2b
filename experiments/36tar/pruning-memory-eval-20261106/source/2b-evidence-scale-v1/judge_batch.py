"""Optional bounded auxiliary evaluation. Never invoked by the GPU worker."""
import argparse,hashlib,json,os,stat,sys,time,urllib.request,urllib.error
from pathlib import Path
R=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(R/'experiments/2b-evidence-demo-v1'))
import payloads as p
import run_judge as old
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def save(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def configure(inputs):
    inputs=inputs.resolve();assert inputs.is_relative_to((R/'runs').resolve())
    p.P=inputs;old.P=inputs
    return inputs
def audit(inputs):
    inputs=configure(inputs);tasks=read(inputs/'tasks.json');cases=read(inputs/'cases_LOCAL_ONLY.json');rep=read(inputs/'repeats_LOCAL_ONLY.json')
    cal=read(R/'runs/evidence_demo_prepare_20261102_01/calibrations_LOCAL_ONLY.json');save(inputs/'calibrations_LOCAL_ONLY.json',cal)
    assert 0<len(tasks)<=45 and len(cases)==5*len(tasks) and len(rep)<=10
    placeholder=dict(requirements=[dict(requirement='x'*150,observed_evidence='x'*300,references=[])]*3,input_coverage='partial',reason='x'*200)
    rows=[]
    for stage,items in [('contract',tasks),('judge',cal+cases+rep)]:
        for item in items:
            request=p.payload(item,stage,None if 'frames' in item else placeholder)
            raw=json.dumps(request,ensure_ascii=False).encode()
            assert not any(x in raw for x in (b'LOCAL_ONLY',b'official_correct',b'expected_fact',b'expected_support',b'heuristic11',b'fixed12',b'mask_ce',b'text_only'))
            rows.append(dict(id=item.get('opaque_id',item['task_id']),stage=stage,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),**p.reserve(request)))
    assert len(rows)<=300
    summary=dict(requests=len(rows),tasks=len(tasks),cases=len(cases),maximum_output_tokens=len(rows)*1024,
        estimated_bytes=sum(x['bytes'] for x in rows),conservative_reservation_cny=sum(x['cny_upper'] for x in rows),
        input_images_verified=True,method_and_label_blind=True,API=0,GPU=0)
    save(inputs/'payload_audit.json',dict(summary=summary,rows=rows));print(json.dumps(summary))
def run(inputs,out):
    inputs=configure(inputs);out=out.resolve();assert out.is_relative_to((R/'runs').resolve()) and not out.exists()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
    auth=read(inputs/'authorization.json');assert auth['approved'] is True
    assert 0<auth['cap_cny']<=25 and 0<auth['max_requests']<=300
    for name,digest in read(inputs/'judge_execution_lock.json').items():
        assert hashlib.sha256((R/name).read_bytes()).hexdigest()==digest,name
    assert hashlib.sha256((inputs/'judge_execution_lock.json').read_bytes()).hexdigest()==auth['lock_sha256']
    auditinfo=read(inputs/'payload_audit.json')['summary'];assert auditinfo['requests']==auth['max_requests']
    secret=R/'.secrets/deepseek.json';assert not secret.is_symlink() and not secret.parent.is_symlink()
    assert secret.stat().st_uid==os.getuid() and stat.S_IMODE(secret.stat().st_mode)==0o600
    key=read(secret)['DEEPSEEK_API_KEY'];assert isinstance(key,str) and key and not any(c.isspace() for c in key)
    sys.path.insert(0,str(R/'scripts'));from deepseek_api_setup import NoRedirect
    opener=urllib.request.build_opener(NoRedirect());out.mkdir();start=time.monotonic();spent=0.;count=uploaded=0;pending=None;results=[];contracts={}
    def ledger():save(out/'budget.json',dict(cap_cny=auth['cap_cny'],peak_upper_cny=spent,requests=count,uploaded_bytes=uploaded,pending=pending))
    def call(item,stage,contract=None):
        nonlocal count,uploaded,spent,pending
        assert time.monotonic()-start<3600 and count<auth['max_requests']
        payload=p.payload(item,stage,contract);reservation=p.reserve(payload);raw=json.dumps(payload,ensure_ascii=False).encode()
        assert spent+reservation['cny_upper']<=auth['cap_cny'] and uploaded+len(raw)<=2*1024**3
        count+=1;uploaded+=len(raw);ident=item.get('opaque_id',item['task_id'])
        pending=dict(id=ident,stage=stage,reservation_cny=reservation['cny_upper']);ledger()
        request=urllib.request.Request('https://api.deepseek.com/chat/completions',data=raw,
            headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'},method='POST')
        with opener.open(request,timeout=90) as resp:body=resp.read(512*1024+1)
        assert len(body)<=512*1024;reply=json.loads(body);usage=reply['usage'];pt=usage['prompt_tokens'];ct=usage['completion_tokens']
        assert type(pt)is int and type(ct)is int and min(pt,ct)>=0
        spent+=pt*2e-6+ct*8e-6;pending=None;ledger()
        choice=reply['choices'][0];text=(choice['message'].get('content') or '').replace(key,'[REDACTED]')
        with (out/'responses.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(dict(id=ident,stage=stage,content=text,usage=usage,model=reply.get('model'),finish_reason=choice.get('finish_reason')),ensure_ascii=False)+'\n')
        assert pt<=reservation['input_token_upper'] and ct<=1024 and spent<=auth['cap_cny']
        assert sum(f.stat().st_size for f in out.iterdir() if f.is_file())<64*2**20
        print(json.dumps(dict(requests=count,cny=spent,id=ident)),flush=True)
        return p.normalize(p.decode_response(text),item,stage)
    try:
        calibration=[]
        for item in read(inputs/'calibrations_LOCAL_ONLY.json'):
            calibration.append(dict(opaque_id=item['opaque_id'],judgment=call(item,'judge')));save(out/'calibration.json',calibration)
        gate=old.calibration_gate(calibration);save(out/'calibration_gate.json',gate)
        if not gate['passed']:
            save(out/'stopped.json',dict(reason='calibration_gate_failed',requests=count,cny=spent));return
        for item in read(inputs/'tasks.json'):
            contracts[item['task_id']]=call(item,'contract');save(out/'contracts.json',contracts)
        for item in read(inputs/'cases_LOCAL_ONLY.json')+read(inputs/'repeats_LOCAL_ONLY.json'):
            results.append(dict(opaque_id=item['opaque_id'],task_id=item['task_id'],repeat_of=item.get('repeat_of'),
                judgment=call(item,'judge',contracts[item['task_id']])));save(out/'judgments.json',results)
        assert count==auth['max_requests']
        save(out/'completed.json',dict(status='completed_auxiliary_not_gold',requests=count,questions=auditinfo['tasks'],explanations=auditinfo['cases'],
            seconds=time.monotonic()-start,peak_upper_cny=spent,GPU=0,training=0,automatic_retry=False))
    except Exception as e:
        ledger();save(out/'failed.json',dict(type=type(e).__name__,http_status=e.code if isinstance(e,urllib.error.HTTPError) else None,
            requests=count,peak_upper_cny=spent,pending=pending,completed_explanations=len(results),automatic_retry=False));raise SystemExit(2)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--inputs',type=Path,required=True);parser.add_argument('--out',type=Path);parser.add_argument('--audit',action='store_true');a=parser.parse_args()
    if a.audit:audit(a.inputs)
    else:
        assert a.out is not None;run(a.inputs,a.out)
