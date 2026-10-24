"""Create E1 in new paths; never alter historical experiments or submit."""
from pathlib import Path
import ast,json,shutil

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'experiments/2b-standalone-cost-v1'
NEW=ROOT/'experiments/2b-auto4-cost-v1'
OUT=ROOT/'runs/2b_auto4_cost_prepare_20261024_01'

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,x):
    with p.open('x',encoding='utf-8',newline='\n') as f:
        f.write(x if isinstance(x,str) else json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def edit(text,a,b):
    assert a in text,a
    return text.replace(a,b)

def main():
    NEW.mkdir(exist_ok=False);OUT.mkdir(exist_ok=False);(OUT/'inputs').mkdir()
    provenance={}
    for name in ('common.py','banks.py','session.py','worker.py','orchestrate.py',
                 'preflight.py','tiny_session_check.py','analyze.py'):
        provenance[name]=str((OLD/name).relative_to(ROOT))
        text=(OLD/name).read_text(encoding='utf-8')
        if name=='common.py':
            text=edit(text,'2b_standalone_cost_prepare_20261023_01','2b_auto4_cost_prepare_20261024_01')
            text+='\nCONFIGS={"slice1":{"thread_count":1,"thread_type":"SLICE"},"auto4":{"thread_count":4,"thread_type":"AUTO"}}\n'
        if name=='session.py':
            text=edit(text,'result = emit_decode(branch, logits, self.builder.processor, 8, stops, on_token)',
                'first_token = []\n            def emit(item):\n                if not first_token:first_token.append(time.monotonic()-started)\n                on_token(item)\n            result = emit_decode(branch, logits, self.builder.processor, 8, stops, emit)')
            text=edit(text,'nodes=meta[\'nodes\'], context_build_seconds=',
                'first_token_seconds=first_token[0], nodes=meta[\'nodes\'], context_build_seconds=')
        if name=='worker.py':
            text=edit(text,'from source import ExactPackets','from source import ExactPackets\n    from io_source import BoundedAVSource')
            text=edit(text,'source=VideoSource(path,dict(source_path=group[\'source_path\'],frames=group[\'frames\'],media_sha256=group[\'sha256\'],duration_s=group[\'duration_s\']),env[\'FramePacket\'])',
                'source=BoundedAVSource(path,group,env[\'FramePacket\'],config=\'auto4\',guard=guard)')
            a=text.index('                    source_cost=dict(decode_seconds=source.decode_seconds')
            b=text.index('\n                else:',a)
            text=text[:a]+'                    source_cost=source.status()'+text[b:]
            text=edit(text,"'QA_total_seconds','QA_cache_bytes'","'QA_total_seconds','QA_cache_bytes','first_token_seconds'")
            text=edit(text,'time.monotonic()-started<1800','time.monotonic()-started<3300')
        if name=='orchestrate.py':
            for a,b in [('4200-','3300-'),('len(results)==9','len(results)==6'),('==909','==228'),
                        ('==11943',"==read(run/'protocol.json')['measured_visual_encodings']"),
                        ('==72','==48'),('<=7344','<=1872'),("['pretrained_loads'] for r in results)==9","['pretrained_loads'] for r in results)==6"),
                        ('workers=9,measured_answers=909,warmup_answers=9','workers=6,measured_answers=228,warmup_answers=6'),
                        ('measured_encodings=11943,warmup_encodings=72,pretrained_loads=9',"measured_encodings=read(run/'protocol.json')['measured_visual_encodings'],warmup_encodings=48,pretrained_loads=6")]:
                text=edit(text,a,b)
        if name=='preflight.py':
            text=edit(text,'len(jobs)==9 and sum(j[\'questions\'] for j in jobs)==909','len(jobs)==6 and sum(j[\'questions\'] for j in jobs)==228')
            text=edit(text,"sorted(g for j in jobs if j['arm']==arm for g in j['groups'])==sorted(g['id'] for g in groups)","sorted(g for j in jobs if j['arm']==arm for g in j['groups'])==sorted(g['id'] for g in groups)*2" )
            # Sort both repeated sequences globally, not concatenate sorted sources.
            text=text.replace("==sorted(g['id'] for g in groups)*2","==sorted([g['id'] for g in groups]*2)")
            text=text.replace('len(layouts)==909','len(layouts)==114').replace('layouts=909','layouts=114')
            text=text.replace('all112_groups_once_per_arm=True,all9_jobs_checked=True','all8_groups_twice_per_arm=True,all6_jobs_checked=True')
        if name=='analyze.py':text=text.replace('expected_answers=909','expected_answers=114').replace('expected=909','expected=114')
        ast.parse(text);write(NEW/name,text)
    write(NEW/'io_source.py',(ROOT/'experiments/video-input-cpu-v1/source.py').read_text(encoding='utf-8'))
    write(NEW/'source.py','''"""V0 auto4 packets; prefix sources stop at their explicit last frame."""
from io_source import BoundedAVSource
class ExactPackets(BoundedAVSource):
    def __init__(self,env,path,frames,digest):
        super().__init__(path,dict(cohort='prefix',frames=frames,sha256=digest),env['FramePacket'],config='auto4')
''')
    origin=ROOT/'runs/video_input_cpu_prepare_20261024_01/inputs'
    for name in ('groups.json','questions.json','snapshots34.json'):
        shutil.copyfile(origin/name,OUT/'inputs'/name)
    groups=read(OUT/'inputs/groups.json');questions=read(OUT/'inputs/questions.json')
    assert len(groups)==8 and len(questions)==38
    ids={q['id'] for q in questions};arms=('full48_8','odd24_8','change24_8')
    labels=read(ROOT/'runs/2b_standalone_cost_prepare_20261023_01/labels_LOCAL_ONLY.json')
    write(OUT/'labels_LOCAL_ONLY.json',[r for r in labels if r['id'] in ids])
    refs=read(ROOT/'runs/2b_standalone_cost_prepare_20261023_01/historical_references_LOCAL_ONLY.json')
    refs['records']=[r for r in refs['records'] if r['id'] in ids and r['arm'] in arms]
    assert len(refs['records'])==114;write(OUT/'historical_references_LOCAL_ONLY.json',refs)
    jobs=[dict(block=rep,arm=arm,groups=[g['id'] for g in groups],questions=38)
          for rep,order in enumerate((arms,arms[::-1])) for arm in order]
    write(OUT/'inputs/jobs.json',jobs)
    per_run=sum((len(g['frames'])+1)//2 for g in groups)
    assert per_run*6+48<=10000
    protocol=dict(arms=list(arms),questions=38,groups=8,measured_answers=228,warmup_answers=6,
        answers_max=234,generated_tokens_max=1872,measured_visual_encodings=per_run*6,
        warmup_visual_encodings=48,visual_encodings=per_run*6+48,pretrained_loads=6,
        warmup_group=next(g['id'] for g in groups if len(g['frames'])>=16),warmup_frames=16,
        decoder='auto4',repeats=2,independent_processes=6,max_new_tokens=8,dtype='FP32',
        max_stream_visual_tokens=384,not_wallclock_1x=True,no_parameter_sweep=True,
        budget=dict(partition='g078t',comment='gx_llm',GPUs=1,CPUs=4,RAM_GiB=24,minutes=60))
    write(OUT/'inputs/protocol.json',protocol)
    write(OUT/'build_provenance.json',dict(base_files=provenance,input_source=str(origin.relative_to(ROOT)),
        old_paths_modified=False,remote_deployed=False,GPU_run=False))
    shell=(OLD/'run_eval.sh').read_text(encoding='utf-8').replace('2b-standalone-cost','2b-auto4-cost').replace('01:15:00','01:00:00').replace('2b_standalone_cost_prepare_20261023_01','2b_auto4_cost_prepare_20261024_01')
    write(NEW/'run_eval.sh',shell)
    print(dict(stage='E1',groups=8,questions=38,jobs=6,answers=228,visual_calls=protocol['visual_encodings']))

if __name__=='__main__':main()
