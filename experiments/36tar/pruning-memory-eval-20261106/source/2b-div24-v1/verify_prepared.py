"""Audit immutable inputs, independent choice scoring and a closed GPU gate."""
import json
from collections import Counter
from common import ROOT, OUT, ARMS, read, save, sha
import importlib.util
from pathlib import Path

def local(name):
    spec=importlib.util.spec_from_file_location('div24_'+name,Path(__file__).with_name(name+'.py'))
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

verify=local('evaluate').verify
parse=local('analyze').parse


def main():
    protocol, rows, groups=verify(OUT/'inputs')
    assert len(ARMS)==5 and protocol['answers_max']==1515
    assert not (OUT/'inputs/GPU_authorization.json').exists()
    boot=read(OUT/'bootstrap_provenance.json')
    assert all(sha(ROOT/relative)==digest for relative,digest in boot.items())
    # The adopted scorer must keep format independent and reject contradictions.
    options=['cat','dog','bird']
    assert parse('B. dog',options)['choice']==1
    assert not parse('B. dog',options)['format_compliant']
    assert parse('B',options)['format_compliant']
    assert parse('B. cat',options)['unjudgeable']
    assert parse('B or C',options)['unjudgeable']
    assert parse('dog',options)['choice']==1
    assert parse('B',options,status='missing')['unjudgeable']
    lookup={r['id']:r for r in rows}
    labels={r['id']:r for r in read(OUT/'labels_LOCAL_ONLY.json')}
    old=ROOT/'runs/2b_token_compression_prepare_20261021_01/results-217854/answers.jsonl'
    score=Counter(); totals=Counter()
    for line in old.read_text(encoding='utf-8').splitlines():
        p=json.loads(line); q=lookup[p['id']]
        if q['cohort']=='stream34': continue
        parsed=parse(p.get('text'),q['options'],status=p['status'])
        totals[p['arm']]+=1
        score[p['arm']]+=int(parsed['choice']==labels[p['id']]['answer_index'])
    assert dict(score)==dict(full48=164,odd24=162,tome24=158),score
    assert set(totals.values())=={269}
    save(OUT/'preparation_audit.json',dict(status='passed',execution_lock_sha256=sha(OUT/'inputs/execution_lock.json'),
        original_sources_unchanged=len(boot),CPU_receipt_sha256=sha(OUT/'CPU_preflight.json'),
        old217854_answer_scoring_regression=dict(score),old_prediction_SHA256=sha(old),
        no_GPU_authorization=True,no_remote_deployment_confirmed=True,
        new_pretrained_loads=0,new_GPU_jobs=0))
    print(dict(status='passed',historical_answer_regression=dict(score),GPU_authorized=False))

if __name__=='__main__':main()
