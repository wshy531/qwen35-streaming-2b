"""Verify collected CPU evidence using the standard library; no model imports."""
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
job=sys.argv[1]
if not job.isdigit():raise ValueError('numeric job ID required')
run=ROOT/('runs/ls-cpu-'+job)
plan=json.loads((ROOT/'reports/long-short-cpu-plan-20261021-01.json').read_text(encoding='utf-8'))
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

states=[]
for p in sorted((ROOT/'logs').glob('cpu-release-20261021-ls-watch-*.json')):
    outer=read(p)
    if outer['returncode']!=0:continue
    for line in outer['stdout'].splitlines():
        row=json.loads(line)
        if row.get('kind')=='command' and row.get('args',[None])[0]=='sacct':
            for entry in row['stdout'].splitlines():
                fields=entry.split('|')
                if fields[0]==job:states.append(fields)
if not states or states[-1][2:4]!=['COMPLETED','0:0']:
    raise ValueError('Slurm completion with exit 0:0 must be observed before acceptance')
state=states[-1]

proof=read(run/'cpu/preflight.json')
protocol=read(run/'protocol.json')
if proof['status']!='passed' or proof['pretrained_loads']!=0:
    raise ValueError('CPU preflight did not pass without real weights')
if protocol['schema']!='online-memory-2b-long-short-v1' or protocol['arms']!=[
    'M0','M2','M2_A06','LS_shared384_v1','reservoir4','text_only']:
    raise ValueError('wrong evaluation profile')
if proof['protocol_sha256']!=sha(run/'protocol.json') or proof['source_lock_sha256']!=sha(run/'source_lock.json'):
    raise ValueError('CPU proof identity mismatch')
source=read(run/'source_lock.json')
for rel,expected in source.items():
    if sha(Path(plan['local_repo'])/rel)!=expected:raise ValueError('local/executed source mismatch: '+rel)
if (proof['inherited_tests_discovered'],proof['inherited_tests'],proof['inherited_tests_skipped'])!=(209,208,1):
    raise ValueError('original regression accounting drift')
inherited=read(run/'cpu/inherited.json')
if (proof['inherited_receipt_sha256']!=sha(run/'cpu/inherited.json')
        or proof['inherited_log_sha256']!=sha(run/'cpu/inherited.txt')
        or inherited['log_sha256']!=sha(run/'cpu/inherited.txt')
        or inherited['source_lock_sha256']!=sha(run/'source_lock.json')):
    raise ValueError('inherited evidence mismatch')
suites={'contracts.txt':28,'tests_a06.txt':16,'tests_ls.txt':25,'long_stream_audit_contracts.txt':8}
for name,count in suites.items():
    text=(run/'cpu'/name).read_text(encoding='utf-8')
    counts=[int(n) for n in re.findall(r'^Ran (\d+) tests? in ',text,re.M)]
    if counts!=[count] or not re.search(r'^OK\s*$',text,re.M) or re.search(r'^FAILED|^OK \(',text,re.M):
        raise ValueError('incomplete or skipped contract suite: '+name)
extra=read(run/'cpu/comparison_contracts.json')
if (extra['source_lock_sha256']!=sha(run/'source_lock.json')
        or extra['protocol_sha256']!=sha(run/'protocol.json')
        or extra['suites']!={folder:dict(tests=count,sha256=sha(run/'cpu'/(folder+'.txt')))
                            for folder,count in (('tests_a06',16),('tests_ls',25))}):
    raise ValueError('comparison contract identity mismatch')
if proof['contracts_log_sha256']!=sha(run/'cpu/contracts.txt'):
    raise ValueError('original contract identity mismatch')
if (proof['media_files_checked'],proof['suffix_questions_checked'],
        proof['source_frames_checked'],proof['source_pairs_checked'])!=(67,171,2766,1402):
    raise ValueError('actual media/input count drift')
if [r['arm'] for r in proof['tiny_answers']]!=protocol['arms'] or any(
    type(r['generated']) is not int or not 0<r['generated']<=8 for r in proof['tiny_answers']):
    raise ValueError('missing six-arm tiny QA integration')
if proof['max_suffix_tokens']>128:raise ValueError('question truncation bound exceeded')
files={p.relative_to(run).as_posix():sha(p) for p in sorted(run.rglob('*')) if p.is_file()}
result=dict(job_id=int(job),status='CPU_preflight_independently_verified',
    slurm_state=state[2],exit_code=state[3],elapsed=state[4],allocated_TRES=state[7],
    inherited_discovered=209,inherited_passed=208,platform_not_applicable=proof['inherited_platform_skips'],
    online_passed=28,A06_passed=16,LS_passed=25,audit_passed=8,
    total_discovered=286,total_applicable_passed=285,
    sources_RGB_PTS_checked=67,sampled_frames=2766,frame_pairs=1402,
    complete_questions_checked=171,maximum_question_suffix_tokens=proof['max_suffix_tokens'],
    tiny_random_QA=proof['tiny_answers'],real_2B_weight_loads=0,
    protocol_sha256=sha(run/'protocol.json'),preflight_sha256=sha(run/'cpu/preflight.json'),
    environment=proof['environment'],source_matches_local=True,artifacts_sha256=files,
    GPU_capability_evaluation_executed=False,accuracy_gain_established=False,
    long_video_34_inputs_preflighted=False)
out=ROOT/('reports/job-'+job+'-LS-CPU-result-20261021.json')
with out.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps(dict(report=str(out),status=result['status'],passed=285,discovered=286,real_weights=0)))
