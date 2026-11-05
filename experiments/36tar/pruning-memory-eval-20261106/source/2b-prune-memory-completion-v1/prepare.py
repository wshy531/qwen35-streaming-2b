"""Freeze existing panels and exact old reference rows. No model invocation."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/prune_memory_completion_prepare_20261105_01'
P=ROOT/'runs/midprune_memory_prepare_20261102_01'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    OUT.mkdir(exist_ok=True)
    stream=read(ROOT/'runs/stream_prune_repair_prepare_20261104_01/questions.json')
    paired=read(P/'questions.json')
    assert len(stream)==75 and len(paired)==18
    assert len({q['source_id'] for q in stream})==72
    sources={q['source_id']:dict(name=q['source_id'],media_path=q['media_path'],media_sha256=q['media_sha256']) for q in stream}
    save(OUT/'stream_questions.json',stream)
    save(OUT/'stream_sources.json',[sources[k] for k in sorted(sources)])
    save(OUT/'paired_questions.json',paired)
    save(OUT/'paired_sources.json',read(P/'sources.json'))
    wanted={q['id'] for q in stream}
    refs=[]
    original_paths=[ROOT/'runs/memory-prune-panel150-220382/answers.jsonl',ROOT/'runs/midprune-memory-220237/answers.jsonl']
    for line in original_paths[0].open(encoding='utf-8'):
        r=json.loads(line)
        if r['id'] in wanted and r['phase']=='main' and r['arm'] in ('full/M0','full/LS'):
            arm='full_recent8' if r['arm']=='full/M0' else 'full_LS'
            refs.append(dict(r,panel='stream',arm=arm))
    assert len(refs)==150
    mapping={'full/M0':'full_recent8','full/LS':'full_LS',
             'fixed12/M0_same8':'fixed12_recent8','fixed12/LS':'fixed12_LS',
             'mask_ce/M0_same8':'old_CE_recent8','mask_ce/LS':'old_CE_LS'}
    for line in original_paths[1].open(encoding='utf-8'):
        r=json.loads(line)
        key=r['view']+'/'+r['policy']
        if key in mapping:refs.append(dict(r,panel='paired',arm=mapping[key]))
    assert len(refs)==258 and len({(r['panel'],r['id'],r['arm']) for r in refs})==258
    save(OUT/'references.json',refs)
    plan={'stream':dict(sources=72,questions=75,answers=450,new_answers=300,reference_matches=150,
            pairs=13242,vision=39726,max_generated_tokens=450*128,GPU_minutes=180,system_memory_GiB=48),
          'paired':dict(sources=5,questions=18,answers=180,new_answers=72,reference_matches=108,
            pairs=429,vision=2145,max_generated_tokens=180*128,GPU_minutes=30,system_memory_GiB=24),
          'notes':['Complete history is all sampled causal pairs, not LS. No silent truncation.',
                   'Adaptive event rule inherited; no fixed-three-pair replacement.',
                   'Paired panel has 12 current and 6 old development questions, not an independent benchmark.',
                   'All outputs include B explanations; primary scoring is option content, not format.',
                   'No API, training, download, retry or requeue. Two sequential allocations proposed.']}
    save(OUT/'plan.json',plan)
    save(OUT/'reference_receipt.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in original_paths})
    print(json.dumps(plan))

if __name__=='__main__':main()
