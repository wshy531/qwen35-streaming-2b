"""CPU-only independent audit of the downloaded approved 2B job artifacts."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/first_2b_gpu_217135_20261020_01'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
fetch=read(OUT/'fetch_receipt.json')
for name,info in fetch['files'].items():assert (OUT/name).stat().st_size==info['bytes'] and sha(OUT/name)==info['sha256']
original=read(OUT/'original/summary.json');stream=read(OUT/'stream/summary.json');remote=read(OUT/'comparison/comparison.json')
assert original['status']==stream['status']=='complete' and original['stage']==stream['stage']=='gpu'
assert original['pretrained_weights_loaded'] and stream['pretrained_weights_loaded']
assert len(original['records'])==16 and len(stream['records'])==8
assert original['identity']['script_sha256']==stream['identity']['script_sha256']==sha(OUT/'source_snapshot/first_2b_baseline.py')
assert sha(OUT/'source_snapshot/first_2b_baseline.py')==sha(ROOT/'experiments/2b-memory-eval/first_2b_baseline.py')
assert sha(OUT/'source_snapshot/analyze_first_baseline.py')==sha(ROOT/'experiments/2b-memory-eval/analyze_first_baseline.py')
assert sha(OUT/'source_snapshot/run_first_2b_baseline.sh')=='21731748f40182c80ebba78e4b1ac7ba4942d5590f60c3c65882cd93cd1a91c8'
for key in ['script_sha256','model_manifest_sha256','online_source_sha256','hybrid_source_sha256','source_sha256','question','canvas','max_new_tokens']:
    assert original['identity'][key]==stream['identity'][key],key
proof=read(ROOT/'runs/first_2b_submission_20261020_01/preflight_stdout.json')
for report,kind in [(original,'original'),(stream,'stream')]:
    assert report['identity']==next(p['identity'] for p in proof['proofs'] if p['implementation']==kind)
guards={stage:read(OUT/stage/'resource_guard.json') for stage in ['original','stream']}
for stage,g in guards.items():assert g['slurm_job_id']=='217135' and g['free_gpu_bytes']>=12*2**30 and g['available_ram_bytes']>=12*2**30
results=[];all_rows=original['records']+stream['records'];tokens=sum(len(r['generated_ids']) for r in all_rows)
assert tokens<=384 and all(0<len(r['generated_ids'])<=16 for r in all_rows)
assert len({(r['source'],r['prefix_frames'],r['mode']) for r in all_rows})==24
for s in stream['records']:
    matched=next(r for r in original['records'] if (r['source'],r['prefix_frames'],r['mode'])==(s['source'],s['prefix_frames'],'matched'))
    native=next(r for r in original['records'] if (r['source'],r['prefix_frames'],r['mode'])==(s['source'],s['prefix_frames'],'native'))
    assert matched['frames']==native['frames']==s['frames'] and len(s['frames'])==s['prefix_frames']
    assert matched['protocol']['input_ids']==s['input_ids']
    assert s['main_unchanged'] and matched['generated_ids']==s['generated_ids']
    x=np.load(OUT/'original'/matched['logits_file'],allow_pickle=False)
    y=np.load(OUT/'stream'/s['logits_file'],allow_pickle=False)
    z=np.load(OUT/'original'/native['logits_file'],allow_pickle=False)
    assert x.shape==y.shape==z.shape and x.dtype==y.dtype==z.dtype==np.dtype('float32')
    assert np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(z).all()
    assert np.allclose(x,y,atol=1e-4,rtol=1e-4)
    error=float(np.max(np.abs(x-y)))
    logged=next(r for r in remote['cases'] if (r['source'],r['prefix_frames'])==(s['source'],s['prefix_frames']))
    assert error==logged['max_abs_logit_error']
    results.append(dict(source=s['source'],prefix_frames=s['prefix_frames'],max_abs_error=error,
        numerical_tolerance_pass=True,matched_generated_same=True,native_generated_same=native['generated_ids']==s['generated_ids'],
        native_input_ids_same=native['input_ids']==s['input_ids'],runtime_main_state_comparison_unchanged=True,
        native_logit_max_abs_error=float(np.max(np.abs(z-y))),native_answer=native['answer'],stream_answer=s['answer']))
assert len(results)==8 and remote['passed']
receipt=dict(status='passed',job_id='217135',method='local NumPy reload of all24 saved FP32 initial logit vectors and direct record/identity comparison',
    matched_cases=8,matched_generated_ids_same=8,native_generated_ids_same=sum(r['native_generated_same'] for r in results),
    max_matched_first_logit_abs_error=max(r['max_abs_error'] for r in results),
    runtime_unchanged_main_state_records=8,main_state_scope='checked runtime flags and pinned executing source; full before/after cache tensors not downloaded',
    actual_answers=24,generated_tokens=tokens,answers_at16_token_cap=sum(len(r['generated_ids'])==16 for r in all_rows),
    real_pretrained_loads=2,peak_GPU_allocated_bytes={k:r['peak_allocated_bytes'] for k,r in [('original',original),('stream',stream)]},
    inference_loop_seconds=dict(original=original['wall_seconds'],stream=stream['wall_seconds']),
    guards=guards,results=results,new_GPU_or_model_execution_in_this_audit=0,
    limits='Four old OVO sources, two/four frames only; no long-memory/reset/compression/semantic accuracy or throughput claim. Original has16 answers, stream8, so loop times are not a matched speed comparison.')
(OUT/'independent_audit.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in receipt.items() if k not in ['guards','results']},ensure_ascii=False,indent=2))
