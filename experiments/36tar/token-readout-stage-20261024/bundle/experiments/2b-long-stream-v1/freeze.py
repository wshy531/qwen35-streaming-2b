"""Freeze a finite long-stream integration comparison, not a final benchmark."""
import ast
import json
from pathlib import Path

from inventory_local import ROOT,OUT
from evaluate import sha,read,write


def main():
    here=Path(__file__).parent;dest=OUT/'runtime_inputs';dest.mkdir(exist_ok=False)
    native=read(OUT/'CPU_native_pixels34_v3.json');memory=read(OUT/'CPU_final_current_code_v2.json')
    assert native['status']==memory['status']=='passed'
    for name in ('native.py',):assert native['runtime_sources_sha256'][name]==sha(here/name)
    for name in ('observer.py','online_bank.py','replay.py','schedule.py'):
        assert memory['runtime_sources_sha256'][name]==sha(here/name),name
    for p in here.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
    sources=read(OUT/'stream_sources6.json')
    media=read(ROOT/'runs/2b_collaboration_sync_20261020_01/media_manifest124.json')
    for s in sources:
        matches=[m for m in media if m['dataset']=='ovo' and m['original_source_path']==s['source_path']]
        assert len(matches)==1 and matches[0]['sha256']==s['media_sha256']
        s['media_path']='data/2b-memory-eval-20261020/'+matches[0]['relative_path']
    write(dest/'sources.json',sources)
    write(dest/'requests.json',read(OUT/'stream_requests34.json'))
    write(dest/'snapshots.json',read(OUT/'stream_query_snapshots34.json'))
    write(dest/'expected_layout34.json',read(OUT/'CPU_long_layout34_v2.json')['contexts'])
    write(dest/'local_cpu_proofs.json',dict(native_passed=True,memory_passed=True,
        native_proof_sha256=sha(OUT/'CPU_native_pixels34_v3.json'),
        memory_proof_sha256=sha(OUT/'CPU_final_current_code_v2.json'),
        media_proof_sha256=sha(OUT/'CPU_full_media_replay.json'),
        scopes=dict(native='34 full native pixel/layout checks; six short full/chunk bitwise equal; one tiny original callback',
            memory='24 actual pair encodings; six isolated tiny query branches;102 layouts; final34 metadata dispatch',
            media='2853 actual PTS/RGB frames and34 committed-prefix checks with fake observer; no pretrained model')))
    lock=read(ROOT/'runs/2b_validation_T_prepare_20261020_01/runtime_inputs/runtime_lock.json')
    for name,digest in lock.items():assert sha(ROOT/name)==digest,name
    for p in here.glob('*'):
        if p.suffix in ('.py','.sh'):lock[p.relative_to(ROOT).as_posix()]=sha(p)
    repo=ROOT/'publish/qwen35-streaming-2b'
    for folder in (repo/'qwen35_local',repo/'experiments/36tar/streaming-learning/code/qwen35-streaming-s4'):
        for p in folder.rglob('*.py'):lock['repo/'+p.relative_to(repo).as_posix()]=sha(p)
    model=ROOT/'models/Qwen3.5-2B'
    for p in model.iterdir():
        if p.is_file() and p.stat().st_size<20*2**20:lock[p.relative_to(ROOT).as_posix()]=sha(p)
    write(dest/'runtime_lock.json',lock)
    protocol=dict(status='preregistered_development_integration_NOT_final_benchmark',questions=34,sources=6,
        original28_preserved=True,original_review=dict(supported=16,ambiguous=11,not_established_at_committed_cutoff=1),
        manual6_separate_development=True,arms=['native','main_only','M0','M2','text_only'],
        pairing='complete_pair_only_no_query_padding',sampling_fps=1,canvas_hw=[192,256],
        playback='causal_replay_NOT_wallclock',generation='incremental_callback',max_new_tokens=8,
        main_context_tokens=2048,question_reserve=128,bank_visual_tokens=384,bank_max_nodes=8,
        native_context_guard=32768,native_config_max_position_embeddings=262144,
        native_pixel_preprocess='pair sized, exact ordered concat, original whole visual/language forward',
        pretrained_loads_max=2,answers_max=170,generated_tokens_max=1360,
        partition='g078t',comment='gx_llm',GPUs=1,CPUs=4,host_memory_GiB=24,wall_seconds=3600,
        worker_native_seconds=900,worker_memory_seconds=2400,
        peak_GPU_allocated_guard_GiB=60,process_RSS_observed_guard_GiB=20,logs_cap_bytes_per_worker=64*2**20,
        source_seconds_range=[min(s['duration_s'] for s in sources),max(s['duration_s'] for s in sources)],
        input_sha256={n:sha(dest/n) for n in ('sources.json','requests.json','snapshots.json','expected_layout34.json','local_cpu_proofs.json','runtime_lock.json')},
        fairness='same arrived complete real frames, exact questions/options/QA times; bank arms use only own bank for history answers',
        caveats=['six reused sources, single reviewer; not unseen benchmark','main_only has different access and reset semantics',
          'native retains full RGB with all reread costs; candidate has no full feature/RGB archive',
          'text side-control runs in M0 query branch; do not compare overall arm wall time as pure speed',
          'late-cue/removal necessity not yet demonstrated; no architecture/prompt sweep'],
        analysis='all28 original kept; supported16 and manual6 separate; retain errors/format/missing; paired corrections/losses and per-source results',
        stop='one fixed run; no automatic retry, prompt/threshold search or new architecture; review then required1x playback stage',
        GPU_authorized=False,downloads=0,training=0)
    write(dest/'protocol.json',protocol)
    print(dict(files=len(list(dest.iterdir())),runtime_sources=len(lock),answers=170,pretrained_loads_max=2,GPU_authorized=False))


if __name__=='__main__':main()
