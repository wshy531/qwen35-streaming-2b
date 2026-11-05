import hashlib,json
from pathlib import Path
from payloads import R,P,read,normalize,decode_response
def main():
    cases=read(P/'cases_LOCAL_ONLY.json');tasks=read(P/'tasks.json')
    assert len(cases)==72 and len(tasks)==24
    assert len({(c['LOCAL_ONLY']['question_id'],c['LOCAL_ONLY']['arm']) for c in cases})==72
    for t in tasks:
        rows=[c for c in cases if c['task_id']==t['task_id']]
        assert {c['LOCAL_ONLY']['arm'] for c in rows}=={'full','fixed12','heuristic11'}
        assert len(t['frames'])<=40
    t=tasks[0];frame=t['frames'][0]['frame_id']
    good=decode_response('```json\n'+json.dumps(dict(support=' Sufficient ',core_error='false',checks=[dict(references=[f'Frame {frame}'])]))+'\n```')
    x=normalize(good,t,'judge');assert x['support']=='sufficient' and x['core_error'] is False and not x['processing_issues']
    x=normalize(dict(support='partial',core_error=False,checks=[dict(references=[999999])]),t,'judge');assert 'unknown_frame' in x['processing_issues']
    x=normalize(decode_response('Not enough information to parse.'),t,'judge');assert x['processing_status']=='needs_review'
    files=[*sorted((R/'experiments/2b-evidence-demo-v1').glob('*.py')),
        *[P/n for n in ('tasks.json','cases_LOCAL_ONLY.json','repeats_LOCAL_ONLY.json','calibrations_LOCAL_ONLY.json','selected_metadata.json','selection.json')]]
    lock={f.relative_to(R).as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    dest=P/'execution_lock.json'
    if dest.exists():assert read(dest)==lock,'frozen files changed'
    else:dest.write_text(json.dumps(lock,indent=2)+'\n')
    print(json.dumps(dict(files=len(lock),lock_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),paired_cases=72,format_checks=3,GPU=0,API=0)))
if __name__=='__main__':main()
