"""Freeze existing requests and a minimal, separately owned 2B adapter bundle."""
import hashlib,json,shutil,ast
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'runs/2b_natural_eval_20261020_01'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    OUT.mkdir(exist_ok=False); provenance={}
    for name in ['capability_pilot.py','fixed_stage2.py','readout.py','memory384.py']:
        src=ROOT/'experiments/08b-memory-eval'/name;dst=HERE/name
        dst.write_bytes(src.read_bytes());provenance[name]=dict(source=str(src.relative_to(ROOT)),original_sha256=sha(src),adaptation='byte-identical helper; old CLI entry points never invoked')
    for name in ['first_2b_baseline.py','audit_sampling.py','memory_policy.py','score_memory_eval.py']:
        src=ROOT/'experiments/2b-memory-eval'/name;dst=HERE/name
        dst.write_bytes(src.read_bytes());provenance[name]=dict(source=str(src.relative_to(ROOT)),original_sha256=sha(src),adaptation='byte-identical helper')
    diag=ROOT/'publish/qwen35-streaming/experiments/36tar/streaming-learning/diagnostics'
    for src,name in [(diag/'visual_memory_v1/features.py','features.py'),(diag/'forest_runtime_v1/forest.py','forest.py')]:
        old=src.read_text(encoding='utf-8');assert old.count('<= 1024')+old.count('<=1024')==1
        new=old.replace('<= 1024','<= 2048').replace('<=1024','<=2048')
        (HERE/name).write_text(new,encoding='utf-8');provenance[name]=dict(source=str(src.relative_to(ROOT)),original_sha256=sha(src),adaptation='validation width bound 1024 -> 2048 only; algorithm unchanged')
    for rel in ['event_memory_v1/memory.py','forest_runtime_v1/cases.json']:
        dst=HERE/'diag'/rel;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes((diag/rel).read_bytes())
        provenance['diag/'+rel]=dict(source=str((diag/rel).relative_to(ROOT)),original_sha256=sha(diag/rel),adaptation='byte-identical')
    groups=[('public74','runs/08b_frozen74_compare_20261019_01/requests.json',74),('open47','runs/08b_holdout47_compare_20261019_01/requests.json',47),('new50','runs/08b_question_readout_new24_20261019_01/requests.json',50)]
    requests=[];media={};identities=[];ids=set()
    for name,path,n in groups:
        source=json.loads((ROOT/path).read_text(encoding='utf-8'));assert len(source)==n
        for r in source:
            assert r['id'] not in ids;ids.add(r['id'])
            visible={k:r['model_visible'][k] for k in ['question','options','query_time_s','frames']}
            assert all(x['pts_s']<=visible['query_time_s']+1e-9 for x in visible['frames'])
            local=ROOT/r['media_path'];h=sha(local);assert h==r['media_sha256']
            remote='data/fixed-natural-eval-v1/'+h[:16]+'.mp4'
            media[remote]=dict(local_path=r['media_path'],remote_path=remote,bytes=local.stat().st_size,sha256=h)
            requests.append(dict(id=r['id'],cohort=name,media_path=remote,media_sha256=h,model_visible=visible))
        identities.append(dict(cohort=name,source_requests=path,source_sha256=sha(ROOT/path),questions=n))
    save(OUT/'requests.json',requests);save(OUT/'media_manifest.json',list(media.values()));save(OUT/'reuse_provenance.json',provenance)
    save(OUT/'protocol.json',dict(status='locked_before_2B_natural_answers',model_id='Qwen/Qwen3.5-2B',revision='15852e8c16360a2fea060d615a32b45270f8a8fc',questions=len(requests),videos=len(media),cohorts=identities,request_sha256=sha(OUT/'requests.json'),precision='float32',canvas=[192,256],max_new_tokens=8,native_context_ceiling=3072,memory_context_ceiling=8192,candidate_visual_ceiling=384,arms=dict(native=['native_original'],memory=['full_pairwise_reference','M0','M1','M2','text_only']),maximum_answers=1026,maximum_generated_tokens=8208,maximum_pretrained_loads=2,worker_seconds=1800,allocated_GPU_ceiling_GiB=24,preload_free_RAM_GiB=12,preload_free_GPU_GiB=16,deadline_UTC='2026-10-20T06:11:50+00:00',stop='fixed rules, no prompt/threshold sweeps, preserve all errors; engineering retries separately recorded',not_independent_unseen_holdout=True,forest_threshold=0.047536224126815796,forest_threshold_transfer_not_optimized_for_2B=True,full_reference_cache_separate_from_candidate_storage=True,media_bytes=sum(x['bytes'] for x in media.values())))
    save(OUT/'source_lock.json',{p.relative_to(HERE).as_posix():sha(p) for p in HERE.rglob('*') if p.is_file() and p.suffix in ['.py','.sh','.json']})
    for p in HERE.rglob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
    print(json.dumps(dict(questions=len(requests),videos=len(media),bytes=sum(x['bytes'] for x in media.values()),source_files=len(json.loads((OUT/'source_lock.json').read_text())))))
if __name__=='__main__':main()
