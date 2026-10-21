"""Accept collected long-video CPU evidence only after independent validation."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'repos/online-memory'))
from experiments.wshy.online_memory_2b.isolation import configure
configure()
from experiments.wshy.online_memory_2b.receipts import inherited_result, successful_counts
from experiments.wshy.long_video_2b.runner import FIXED, SUITES, validate_inputs, runtime_lock


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def require(value,message):
    if not value:
        raise ValueError(message)


def main():
    job = sys.argv[1]
    require(job.isdigit(),'numeric job ID required')
    run = ROOT/('runs/long-video-cpu-'+job)
    states = []
    for path in sorted((ROOT/'logs').glob('cpu-release-20261021-long-video-cpu-watch-*.json')):
        receipt = read(path)
        if receipt['returncode'] != 0:
            continue
        for line in receipt['stdout'].splitlines():
            row = json.loads(line)
            if row.get('kind') == 'command' and row['args'][0] == 'sacct':
                states.extend(entry.split('|') for entry in row['stdout'].splitlines() if entry.split('|')[0]==job)
    require(states and states[-1][2:4]==['COMPLETED','0:0'],'Slurm successful completion required')
    protocol, proof = read(run/'protocol.json'),read(run/'cpu/preflight.json')
    require(all(protocol.get(k)==v for k,v in FIXED.items()),'fixed protocol mismatch')
    require(proof['status']=='passed' and proof['pretrained_loads']==0,'CPU proof did not pass without real weights')
    require(proof['protocol_sha256']==sha(run/'protocol.json')
            and proof['source_lock_sha256']==sha(run/'source_lock.json'),'CPU proof binding drift')
    require(read(run/'source_lock.json')==runtime_lock(),'current/executed runtime mismatch')
    for name,expected in protocol['input_hashes'].items():
        require(sha(run/name)==expected,'prepared input hash changed: '+name)
    queries,sources,snapshots = validate_inputs(run)
    plan = read(ROOT/'reports/long-video-cpu-plan-20261021-01.json')
    for name in ('requests.json','sources.json','snapshots.json','scoring_ONLY.json','receipt.json'):
        require(sha(run/name)==plan['input_files'][name]['sha256'],'approved input mismatch: '+name)
    inherited_path = run/'cpu/inherited.txt'
    require(proof['inherited_log_sha256']==sha(inherited_path),'inherited log hash changed')
    inherited = inherited_result(inherited_path.read_text(encoding='utf-8'),platform='linux')
    require(inherited==proof['inherited'],'inherited counts/applicability mismatch')
    require((inherited['tests_discovered'],inherited['tests_run'],inherited['tests_skipped'])==(209,208,1),
            'original regression denominator')
    require(set(proof['suites'])==set(SUITES),'missing contract suite')
    for folder,count in SUITES.items():
        record = proof['suites'][folder]
        path = run/'cpu'/(folder.replace('/','_')+'.txt')
        require(record==dict(tests=count,sha256=sha(path),file=path.name),'suite proof mismatch')
        successful_counts(path.read_text(encoding='utf-8'),[count])
    require((proof['sources_RGB_PTS_checked'],proof['sampled_frames'],proof['pair_callbacks'],
             proof['actual_visual_model_forwards'],proof['queries_checked'])==(6,2726,1360,0,34),'real input coverage')
    require(proof['max_suffix_tokens']<=128,'question truncation bound exceeded')
    checks_path = run/'cpu/actual_query_checks.json'
    require(sha(checks_path)==proof['actual_query_checks_sha256'],'query evidence hash mismatch')
    checks = read(checks_path)
    expected = {r['query_id']:r for r in snapshots}
    require(len(checks)==34 and {r['id'] for r in checks}==set(expected),'actual query grid')
    for row in checks:
        require(all(row[k]==expected[row['id']][k] for k in row if k!='id'),'actual schedule/historical mismatch')
    tiny,answers = proof['tiny'],proof['tiny_answers']
    require(tiny['queries']==2 and tiny['answers']==12 and tiny['encoded_pairs']==12
            and tiny['generated_tokens']<=96 and tiny['reset_checks']==5
            and tiny['all_QA_banks_unchanged'] and tiny['model_loads_by_adapter']==0,'tiny integration coverage')
    grid = Counter((r['id'],r['arm']) for r in answers)
    require(grid==Counter({('tiny:'+str(i),a):1 for i in range(2) for a in protocol['arms']})
            and all(type(r['generated']) is int and 0<r['generated']<=8 for r in answers)
            and sum(r['generated'] for r in answers)==tiny['generated_tokens'],'tiny answer grid/budget')
    require(proof['independent_video_content_review']=='pending' and proof['long_video_accuracy_evaluated'] is False,
            'unreviewed ability claim')
    result = dict(status='long_video_CPU_preflight_independently_verified',job_id=int(job),
        slurm_state=states[-1][2],exit_code=states[-1][3],elapsed=states[-1][4],allocated_TRES=states[-1][7],
        inherited_discovered=209,inherited_passed=208,platform_NA=inherited['platform_skips'],
        contracts={k:v for k,v in SUITES.items()},total_discovered=312,total_applicable_passed=311,
        real_media=6,sampled_frames=2726,complete_pair_callbacks=1360,actual_visual_model_forwards=0,
        query_snapshots_matched=34,complete_question_suffix_max=proof['max_suffix_tokens'],
        tiny_answers=answers,real_pretrained_loads=0,environment=proof['environment'],
        independent_video_content_review='pending',long_video_accuracy_evaluated=False,
        protocol_sha256=sha(run/'protocol.json'),preflight_sha256=sha(run/'cpu/preflight.json'),
        source_lock_sha256=sha(run/'source_lock.json'),
        artifacts_sha256={p.relative_to(run).as_posix():sha(p) for p in sorted(run.rglob('*')) if p.is_file()})
    target = ROOT/('reports/job-'+job+'-long-video-CPU-result-20261021.json')
    with target.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2)
    print(json.dumps({k:result[k] for k in ('status','job_id','elapsed','total_applicable_passed','sampled_frames',
        'complete_pair_callbacks','query_snapshots_matched','complete_question_suffix_max','real_pretrained_loads')}))


if __name__=='__main__':
    main()
