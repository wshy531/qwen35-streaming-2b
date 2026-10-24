"""Fixed scoring contracts and baseline replay references, without model loading."""
import gzip
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/2b-ls-depth-v1'))
from common import OUT, read, save, sha
from analyze import parse, analyze


def main():
    options = ['red','blue','green']
    for text, choice, compliant in [('A',0,True),('A. because the cloth is red',0,False),
            ('The answer is B',1,False),('blue',1,False),('A. blue',None,False),
            ('A or B',None,False),('A. B. other',None,False),('',None,False),('E',None,True)]:
        result = parse(text,options)
        assert result['choice'] == choice and result['format_compliant'] == compliant,(text,result)
    assert parse('A',options,status='failure')['choice'] is None
    # Every missing answer remains, with separate format and answer measures.
    empty = OUT/'empty_synthetic_predictions'; empty.mkdir(exist_ok=False)
    target = OUT/'synthetic_analysis.json'
    analyze(empty,target)
    result = read(target)
    assert len(result['details']) == 1818 and result['actual_answers'] == 0
    for table in result['tables'].values():
        assert all(s['correct'] == 0 and s['unjudgeable'] == s['denominator'] for s in table['scores'].values())
    target.unlink(); empty.rmdir()
    base = ROOT/'runs/review_packet_20261021_01/partner_evidence/evidence'
    mapping = dict(M0='M0',M2='M2',M2_A06='A06',LS_shared384_v1='LS',reservoir4='reservoir4')
    references, provenance = [], {}
    for folder in ('ls-gpu-217845','long-video-gpu-217852'):
        path = base/folder/'gpu/predictions.jsonl.gz'; provenance[path.relative_to(ROOT).as_posix()] = sha(path)
        for line in gzip.open(path,'rt',encoding='utf-8'):
            row = json.loads(line)
            if row['arm'] in mapping:
                references.append(dict(id=row['id'],arm=mapping[row['arm']],generated_ids=row['generated_ids'],
                    text=row['text'],input_ids_sha256=row.get('input_ids_sha256'),node_digest=row.get('node_digest')))
    assert len(references) == 1025 and len({(r['id'],r['arm']) for r in references}) == 1025
    save(OUT/'historical_references_LOCAL_ONLY.json',dict(records=references,provenance=provenance,
        rule='check prior five bank arms on171+34 before interpreting candidate; T98 A06/LS previously absent'))
    save(OUT/'scorer_contract.json',dict(status='passed',parse_cases=10,synthetic_missing_answers=1818,
        synthetic_artifacts_removed=True,historical_reference_answers=len(references),GPU_submissions=0))
    print(dict(scorer='passed',historical_references=len(references),model_loads=0))


if __name__ == '__main__': main()
