"""Freeze data strata and a bounded first-version protocol without reading new answers."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import bridge

HERE, ROOT, BASE = bridge.HERE, bridge.ROOT, bridge.BASE
OLD = ROOT/'runs/2b_natural_eval_20261020_01'
EARLY = ROOT/'runs/2b_early9_diagnostic_20261020_01'
OUT = ROOT/'runs/2b_short_event_prepare_20261020_01'
ARMS = ['full_pairwise_reference', 'M0', 'M1', 'M2', 'B02_cap4', 'reservoir4', 'text_only']
DIAG_ARMS = ['manual_old4', 'm2_target_roots', 'm2_target_roots_recent4',
             'm2_all_old', 'manual_old4_other_old']

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def save(p, value):
    with Path(p).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')

def main():
    OUT.mkdir(exist_ok=False)
    requests = read(OLD/'requests.json')
    keys = {r['id']: r for r in read(OLD/'scorer_rows_local.json')}
    assert len(requests) == len(keys) == 171
    strata = []
    for r in requests:
        k = keys[r['id']]
        assert r['model_visible']['question'] == k['question']
        assert r['model_visible']['options'] == k['options']
        frames = r['model_visible']['frames']
        assert all(f['pts_s'] <= r['model_visible']['query_time_s']+1e-9 for f in frames)
        recent_start = frames[max(0, len(frames)-16)]['pts_s']
        intervals = k.get('evidence_intervals', [])
        if not intervals:
            coverage = 'unknown'
        elif all(a >= recent_start-1e-9 for a, b in intervals):
            coverage = 'all_by_time_span'
        elif all(b < recent_start-1e-9 for a, b in intervals):
            coverage = 'none_by_time_span'
        else:
            coverage = 'partial_by_time_span'
        strata.append(dict(id=r['id'], cohort=r['cohort'], stratum=k['stratum'],
                           source_group_id=k['source_group_id'],
                           pairs=(len(frames)+1)//2, storage_pressure=(len(frames)+1)//2 > 8,
                           recent8_begin_s=recent_start, query_time_s=r['model_visible']['query_time_s'],
                           author_evidence_recent8=coverage, necessary_old_evidence_proven=False,
                           coverage_is_time_proxy_not_content_proof=True))
    save(OUT/'requests.json', requests)
    save(OUT/'diagnostic_requests.json', read(EARLY/'requests.json'))
    (OUT/'baseline_timelines.json').write_bytes((OLD/'gpu_217141/result/memory/timelines.json').read_bytes())
    save(OUT/'strata_local.json', strata)
    save(OUT/'scorer_rows_LOCAL_ONLY.json', list(keys.values()))
    previous = []
    for stage in ['native','memory']:
        previous += [json.loads(line) for line in (OLD/('gpu_217141/result/'+stage+'/predictions.jsonl')).read_text(encoding='utf-8').splitlines()]
    save(OUT/'previous_predictions_LOCAL_ONLY.json', previous)
    sources = {p.name: sha(p) for p in HERE.glob('*') if p.suffix in ('.py', '.sh', '.md')}
    # The prior deployment contains the locked inference bundle, not local-only analyses.
    dependency = read(OLD/'source_lock.json')
    assert all(sha(BASE/rel) == h for rel,h in dependency.items())
    save(OUT/'source_lock.json', sources)
    save(OUT/'dependency_lock.json', dependency)
    protocol = dict(status='prepared_not_GPU_authorized', candidate='B02_cap4', cap_pairs=4,
                    cap_reason='fixed structural size equals recent4, no parameter sweep',
                    reservoir_seed=20261020, candidate_visual_ceiling=384,
                    recent_pairs=4, closed_ceiling=144, open_ceiling=48,
                    arms=ARMS, diagnostic_arms=DIAG_ARMS, questions=171,
                    diagnostic_questions=9, max_answers=1242, max_generated_tokens=9936,
                    max_pretrained_loads=2, worker_seconds=1500, slurm_minutes=60,
                    gpus=1, cpus=4, memory_GiB=24, partition='g078t', comment='gx_llm',
                    precision='float32', max_new_tokens=8, bootstrap_seed=20261020,
                    bootstrap_replicates=2000, primary='F/T by cohort and source groups',
                    secondary='C; all171 strict/content; metadata pressure/coverage strata',
                    scoring='strict exact option plus frozen leading option; format rate separate',
                    diagnostics='manual localization, not automatic retrieval or new holdout',
                    report_required=['corrections','new_losses','recent_regressions','forced_seals',
                                     'within_and_cross_merges','root_depth','storage_bytes','cost'],
                    request_sha256=sha(OUT/'requests.json'),
                    diagnostic_request_sha256=sha(OUT/'diagnostic_requests.json'),
                    baseline_timelines_sha256=sha(OUT/'baseline_timelines.json'),
                    source_lock_sha256=sha(OUT/'source_lock.json'),
                    dependency_lock_sha256=sha(OUT/'dependency_lock.json'),
                    fixed_threshold=0.047536224126815796,
                    stop='one fixed first version; no prompt/threshold/cap/seed search after results',
                    new_source_validation='T: at least40 unopened source candidates/target120 original questions; acquisition not authorized here',
                    limitations=['all171 opened already','same8P storage, actual read amounts may differ',
                                 'single reservoir seed','author intervals not necessity proof',
                                 'no claim that all merged root depths are capped'])
    save(OUT/'protocol.json', protocol)
    save(OUT/'preparation_receipt.json', dict(questions=171, videos=len({r['media_path'] for r in requests}),
         strata=Counter(s['stratum'] for s in strata),
         evidence_coverage=Counter(s['author_evidence_recent8'] for s in strata),
         pressure_questions=sum(s['storage_pressure'] for s in strata),
         requests_unchanged=sha(OUT/'requests.json') == sha(OLD/'requests.json'),
         note='JSON serialization may differ; all request objects verified identical',
         actual_pretrained_loads=0, actual_GPU_submissions=0))
    print(json.dumps(read(OUT/'preparation_receipt.json'), ensure_ascii=False))

if __name__ == '__main__':
    main()
