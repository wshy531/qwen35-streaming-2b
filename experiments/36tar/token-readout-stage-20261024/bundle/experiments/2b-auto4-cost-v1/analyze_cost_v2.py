"""Audit both complete repeats before reporting paired costs and replay."""
import argparse,json,statistics
from pathlib import Path
from common import OUT,ARMS,read,save,sha
from analyze import analyze

def main(result):
    done=read(result/'completed_reconciled.json');assert done['status']=='reconciled_complete'
    assert done['workers']==6 and done['measured_answers']==228 and done['pretrained_loads']==6
    expected={(r['id'],r['arm']):r for r in read(OUT/'historical_references_LOCAL_ONLY.json')['records']}
    records=done['results'];costs={};regressions=[]
    for rep in range(2):
        target=result/f'repeat-{rep}';target.mkdir(exist_ok=False)
        with (target/'answers.jsonl').open('x',encoding='utf-8') as f:
            for rec in records:
                if rec['completed']['block']!=rep:continue
                rows=[json.loads(x) for x in (result/rec['directory']/'answers.jsonl').read_text(encoding='utf-8').splitlines()]
                events=[json.loads(x) for x in (result/rec['directory']/'tokens.jsonl').read_text(encoding='utf-8').splitlines()]
                assert len(rows)==38 and len({r['id'] for r in rows})==38
                for r in rows:
                    assert r['banks_unchanged'] and r['branch_released']
                    ids=[t['token_id'] for t in events if t['query_id']==r['id']]
                    assert ids==r['generated_ids']
                    old=expected[r['id'],r['arm']]
                    for field in ('generated_ids','input_ids_sha256','node_digest'):
                        if r[field]!=old[field]:regressions.append(dict(repeat=rep,id=r['id'],arm=r['arm'],field=field))
                    f.write(json.dumps(r)+'\n')
        analyze(target,OUT/f'answer_results_repeat{rep}.json')
    for arm in ARMS:
        items=[]
        for rep in range(2):
            rec=next(r for r in records if r['completed']['arm']==arm and r['completed']['block']==rep)
            c=rec['completed'];profiles=c['profiles'];qs=[q for p in profiles for q in p['QA']]
            assert len(profiles)==8 and len(qs)==38
            assert all(p['source']['codec']['requested_threads']==4 and p['source']['completed'] for p in profiles)
            rows={cohort:sum(p['pipeline_wall_seconds'] for p in profiles if p['cohort']==cohort) for cohort in ('natural171','T98','stream34')}
            items.append(dict(repeat=rep,measured_block_wall_seconds=c['measured_block_wall_seconds'],
                source_pipeline_wall_seconds=sum(p['pipeline_wall_seconds'] for p in profiles),cohort_pipeline_seconds=rows,
                QA_total_seconds=sum(q['QA_total_seconds'] for q in qs),
                median_first_token_seconds=statistics.median(q['first_token_seconds'] for q in qs),
                vision_seconds=sum(p['encoding']['vision_seconds'] for p in profiles),
                decode_seconds=sum(p['source']['decode_seconds'] for p in profiles),
                peak_GPU_allocated_bytes=c['measured_peak_GPU_allocated_bytes'],
                peak_GPU_reserved_bytes=c['measured_peak_GPU_reserved_bytes'],
                lifetime_RSS_bytes=c['process_lifetime_peak_RSS_bytes'],
                max_bank_selector_bytes=max(p['bank']['standalone_persistent_tensor_bytes'] for p in profiles),
                max_QA_cache_bytes=max(q['QA_cache_bytes'] for q in qs)))
        costs[arm]=items
    save(OUT/'cost_and_replay_results.json',dict(status='analyzed',execution_status='six workers complete; parent encoding-count assertion failed',original_slurm_status=done['original_slurm_status'],total_visual_calls=done['total_encodings'],costs=costs,replay_mismatches=regressions,
        quality_metric='content correctness; format separate',repeats=2,small_fixed_cost_gate=True,
        limitations=['A800 FP32, causal fast file replay; not edge or paced1x',
          'Two repeats do not establish statistical significance',
          'RSS includes model loading; CUDA peak reset after warmup',
          'No slice1 control: cannot infer V0 cluster gain or interaction',
          'Known compression losses remain;38questions cannot certify quality']))
    assert not regressions,regressions[:5]
    print(dict(status='passed',workers=6,answers=228,replay_mismatches=0))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True)
    main(p.parse_args().results)
