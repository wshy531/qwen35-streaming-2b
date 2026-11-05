"""CPU-only continuation contract; inherited model interfaces plus exact tail keys."""
import os,runpy,py_compile
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
from tail_common import *
if os.name!='nt':os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]))
verify()
m=read(T/'manifest.json');qs={q['id']:q for q in read(P/'questions.json')}
rows=[json.loads(x) for x in (R/'runs'/m['parent']/'answers.jsonl').read_text(encoding='utf-8').splitlines()]
assert sha(R/'runs'/m['parent']/'answers.jsonl')==m['parent_answers_sha256']
keys={(x['id'],x['arm']) for x in rows}
missing={(q,a) for q in qs for a in ARMS}-keys
assert missing=={(q,a) for q in m['missing_ids'] for a in ARMS} and len(missing)==68
assert all(q==qs[q['id']] for q in m['questions'])
assert m['control_id'] not in m['missing_ids']
assert all(x['history_nodes']>0 for x in m['expected_control'].values())
for name in ('worker_tail.py','tail_common.py','prepare_tail.py','check_tail.py'):
    py_compile.compile(str(Path(__file__).parent/name),doraise=True)
# Repeat inherited tiny-model interface check, including all 72 selected layouts.
source=(Path(__file__).parent/'check4.py').read_text(encoding='utf-8')
old_save="save(P/('CPU4_local.json' if os.name=='nt' else 'CPU4_cluster.json'),result)"
assert source.count(old_save)==1
# Original check uses exclusive output creation: compare instead of overwriting its receipt.
source=source.replace(old_save,"assert read(P/('CPU4_local.json' if os.name=='nt' else 'CPU4_cluster.json'))==result")
exec(compile(source,str(Path(__file__).parent/'check4.py'),'exec'),{'__name__':'__main__'})
old=read(P/('CPU4_local.json' if os.name=='nt' else 'CPU4_cluster.json'))
result=dict(passed=True,tail_layouts=72,new_answers=68,replays=4,inherited_600_layouts_sha256=old['layout_sha256'],tiny_answers=old['tiny_answers'],manifest_sha256=sha(T/'manifest.json'),GPU=0)
save(T/('CPU_local.json' if os.name=='nt' else 'CPU_cluster.json'),result)
if os.name!='nt':
    media={q['media_path']:q['media_sha256'] for q in m['questions']}
    for n,h in media.items():assert sha(R/n)==h,n
    save(T/'media_verified.json',dict(passed=True,files=len(media),sha_verified=True))
print(json.dumps(result),flush=True)
