"""Independent post-run counters, source workload and raw-result audit."""
from collections import Counter,defaultdict
import hashlib,json
from common import OUT,ARMS,read,save


def lines(path):
    with path.open(encoding='utf-8') as stream:
        for line in stream:yield json.loads(line)


def main():
    raw=OUT/'results-218160';manifest=read(OUT/'results_218160_manifest.json')
    assert len(manifest)==48
    assert {p.relative_to(raw).as_posix() for p in raw.rglob('*') if p.is_file()}==set(manifest)
    for name,meta in manifest.items():
        path=raw/name
        assert path.stat().st_size==meta['bytes']
        with path.open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==meta['sha256']
    complete=read(raw/'completed.json');assert complete['workers']==9 and complete['pretrained_loads']==9
    assert complete['execution_lock_sha256']=='2a5261e8e80d93e552cf6fccab41009ce8e85a27f71de944b84ef5186323eee3'
    root_answers=list(lines(raw/'answers.jsonl'));root_tokens=list(lines(raw/'tokens.jsonl'))
    joined_answers=[];joined_tokens=[];warm_tokens=0;measured_encodes=0;loads=0;coverage={a:set() for a in ARMS}
    cohorts=defaultdict(lambda:defaultdict(float));source_profiles={}
    for result in complete['results']:
        child=raw/result['directory'];done=read(child/'completed.json');startup=read(child/'startup.json')
        assert result['completed']==done and result['startup']==startup and done['status']=='complete'
        assert done['no_other_bank'] and done['no_main_language_cache']
        assert startup['warmup_encodings']==8 and startup['warmup_answers']==1
        loads+=startup['pretrained_loads'];warm_tokens+=startup['warmup_generated_tokens']
        actual=list(lines(child/'answers.jsonl'));tokens=list(lines(child/'tokens.jsonl'))
        assert len(actual)==done['measured_answers']
        assert sum(len(r['generated_ids']) for r in actual)==done['measured_generated_tokens']
        joined_answers.extend(actual);joined_tokens.extend(tokens)
        arm=done['arm'];assert all(r['arm']==arm for r in actual)
        for p in done['profiles']:
            assert p['group'] not in coverage[arm];coverage[arm].add(p['group'])
            measured_encodes+=p['vision_encodings']
            source_profiles[p['group'],arm]=p
            assert p['bank']['visual_tokens']<=(384 if arm=='full48_8' else 192)
            if arm!='change24_8':assert p['bank']['standalone_selector_descriptor_bytes']==0
            else:assert p['bank']['standalone_selector_descriptor_bytes']==393216
            c=cohorts[arm,p['cohort']]
            c['pipeline_wall_seconds']+=p['pipeline_wall_seconds']
            c['QA_seconds']+=sum(q['QA_total_seconds'] for q in p['QA'])
            c['decode_seconds']+=p['source']['decode_seconds'];c['questions']+=len(p['QA'])
    assert joined_answers==root_answers and joined_tokens==root_tokens
    assert loads==9 and measured_encodes==11943 and sum(len(r['generated_ids']) for r in root_answers)+warm_tokens==complete['generated_tokens']==1891
    assert all(len(v)==112 for v in coverage.values()) and len(root_answers)==909
    index={};per_arm=Counter()
    for r in root_answers:
        key=r['id'],r['arm'];assert key not in index;index[key]=r;per_arm[r['arm']]+=1
        assert r['status']=='ok' and r['banks_unchanged'] and r['branch_released'] and r['no_main_language_cache']
        assert r['input_tokens']+8<=2048 and len(r['generated_ids'])<=8
    assert set(per_arm.values())=={303}
    ts=defaultdict(list)
    for token in root_tokens:ts[token['query_id'],token['arm']].append(token)
    assert set(ts)==set(index)
    for key,r in index.items():
        assert [t['token_id'] for t in ts[key]]==r['generated_ids']
        assert [t['token_index'] for t in ts[key]]==list(range(len(ts[key])))
    for gid in coverage[ARMS[0]]:
        profiles=[source_profiles[gid,a] for a in ARMS]
        assert len({p['source']['decoded_frames'] for p in profiles})==1
        assert len({p['source']['sampled'] for p in profiles})==1
        assert len({p['vision_encodings'] for p in profiles})==1
    replay=read(OUT/'answer_results.json')['historical_replay']
    assert replay['expected']==909 and not any(replay[k] for k in ('generated_ID_mismatches','input_ID_mismatches','node_mismatches'))
    save(OUT/'result_audit.json',dict(status='pass',artifact_files=48,answers=909,pretrained_loads=9,
        measured_encodings=11943,warmup_encodings=72,warmup_answers=9,total_generated_tokens=1891,
        old_rows_reproduced=909,one_bank_only=True,same_per_source_decode_and_encode_counts=True,
        cohort_costs=[dict(arm=arm,cohort=cohort,**v) for (arm,cohort),v in cohorts.items()],
        limitations=['One workload pass per arm, not three full repetitions.',
          'A800 FP32 file replay, no edge hardware or paced1x test.',
          'RSS high-water includes startup/warmup; do not claim runtime host-RAM reduction.']))
    print(dict(status='pass',files=48,answers=909,old_rows=909,total_generated=1891))


if __name__=='__main__':main()
