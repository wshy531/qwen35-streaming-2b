"""Post-run integrity and matched-evidence checks, without model execution."""
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
from common import OUT,ARMS,read,save

RAW=OUT/'results-218063'
EXPECTED={
 'answers.jsonl':'9fdfc531aa1c3e096282f1ecccdb3efd83c98b047e90a78b4b124b1bf8d83aba',
 'events.jsonl':'52685fb39d941dc9697d8e05a9947233a9af02bf312dd5bcfbe5fa9f2cf93477',
 'tokens.jsonl':'12032ae00cfafb6fe48c6b9c0d7c95a149e72f2404c5250d307cc036b136ca2f',
 'loaded.json':'fc451e2f52b3055cac2d6480a706843ad4aa93a4971c8e3f9f1486aa96c47c41',
 'completed.json':'dde57a10757979552e48c278f944c80a384822063d2e16c911f73297d8ec9852'}


def lines(name):
    with (RAW/name).open(encoding='utf-8') as stream:
        for line in stream:yield json.loads(line)


def main():
    hashes={}
    for name,digest in EXPECTED.items():
        with (RAW/name).open('rb') as stream:hashes[name]=hashlib.file_digest(stream,'sha256').hexdigest()
        assert hashes[name]==digest,name
    complete=json.loads((RAW/'completed.json').read_text(encoding='utf-8'))
    assert complete['execution_lock_sha256']=='4e569fa6b45e803b21fbfe6fec13f164b14c04fa437731f8f38daf59870f81b8'
    answers={};per_arm=Counter()
    for r in lines('answers.jsonl'):
        key=r['id'],r['arm'];assert key not in answers
        answers[key]=r;per_arm[r['arm']]+=1
        assert r['status']=='ok' and r['banks_unchanged'] and r['branch_released'] and r['no_main_language_cache']
        assert r['visual_tokens']==sum(n['tokens'] for n in r['nodes'])<=384
        assert r['input_tokens']+8<=2048 and len(r['generated_ids'])<=8
        assert len(r['nodes'])<=int(r['arm'].split('_')[1])
    assert len(answers)==complete['answers']==1515
    assert set(per_arm)==set(ARMS) and set(per_arm.values())=={303}
    geometry=lambda r:[tuple(n[k] for k in ('first','last','begin','end','center')) for n in r['nodes']]
    for q in read(OUT/'inputs/questions.json'):
        a={arm:answers[q['id'],arm] for arm in ARMS}
        assert geometry(a['full48_8'])==geometry(a['odd24_8'])==geometry(a['change24_8'])
        assert a['full48_8']['input_tokens']-a['odd24_8']['input_tokens']==24*min(q['pairs'],8)
        assert a['change24_8']['input_tokens']==a['odd24_8']['input_tokens']
        for mode in ('odd24','change24'):
            assert a[mode+'_8']['nodes']==a[mode+'_16']['nodes'][-8:]
            assert a[mode+'_8']['visual_tokens']<=192
            if q['pairs']<=8:
                for field in ('input_ids_sha256','node_digest','generated_ids'):
                    assert a[mode+'_8'][field]==a[mode+'_16'][field]
    tokens=defaultdict(list)
    for r in lines('tokens.jsonl'):tokens[r['query_id'],r['arm']].append(r)
    assert set(tokens)==set(answers)
    for key,r in answers.items():
        ts=tokens[key]
        assert [t['token_id'] for t in ts]==r['generated_ids']
        assert [t['token_index'] for t in ts]==list(range(len(ts)))
        assert all(a['monotonic_s']<=b['monotonic_s'] for a,b in zip(ts,ts[1:]))
    assert sum(map(len,tokens.values()))==complete['generated_tokens']==3038
    ends={};traces=0;storage={a:0 for a in ARMS}
    for r in lines('events.jsonl'):
        if r.get('kind')=='group_end':
            assert r['group'] not in ends
            ends[r['group']]=r
            for arm,bank in r['state']['banks']['banks'].items():
                storage[arm]=max(storage[arm],bank['standalone_persistent_tensor_bytes'])
        trace=r.get('trace') if r.get('kind') in ('selection','query_only_selection') else r.get('state',{}).get('selection_trace')
        if trace is not None:
            traces+=1
            for ids in trace['selected_indices'].values():
                assert len(ids)==len(set(ids))==24 and ids==sorted(ids) and min(ids)>=0 and max(ids)<48
    assert len(ends)==112 and traces==3981
    assert sum(r['state']['vision_encodings'] for r in ends.values())==complete['vision_encodings']==3981
    scores=read(OUT/'answer_results.json');replay=scores['historical_replay']
    assert replay['expected']==909 and not replay['generated_ID_mismatches'] and not replay['input_ID_mismatches'] and not replay['node_mismatches']
    report=dict(status='pass',SHA256=hashes,answers=1515,per_arm=dict(per_arm),generated_tokens=3038,
                matched_queries=303,selection_trace_count=traces,groups=len(ends),old_M0_replayed_rows=909,
                standalone_persistent_tensor_max_bytes=storage,complete=complete,
                limitations=['Node source metadata checked; feature values/xy exactness covered by CPU and pinned code, not a new GPU tensor replay.',
                             'Shared timings are descriptive; not isolated speed evidence.',
                             'All datasets are opened development data.'])
    save(OUT/'result_audit.json',report)
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
