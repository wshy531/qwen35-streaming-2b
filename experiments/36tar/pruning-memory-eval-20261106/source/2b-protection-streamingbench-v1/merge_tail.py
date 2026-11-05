"""Merge exact missing keys; preserve parent failure and tail provenance."""
import argparse
from tail_common import *

def merge(tail,out):
    m=read(T/'manifest.json');parent=R/'runs'/m['parent']
    assert not out.exists() and out.resolve().is_relative_to((R/'runs').resolve())
    assert read(parent/'download_verified.json')['verified'] and read(tail/'download_verified.json')['verified']
    assert sha(parent/'answers.jsonl')==m['parent_answers_sha256']
    done=read(tail/'completed.json');assert done['scope']=='tail_only' and done['answers']==72 and done['numerical']==4
    old=[json.loads(x) for x in (parent/'answers.jsonl').read_text(encoding='utf-8').splitlines()]
    new=[json.loads(x) for x in (tail/'answers.jsonl').read_text(encoding='utf-8').splitlines()]
    ix={(x['id'],x['arm']):x for x in old};seen=set();appended=[]
    for row in new:
        key=(row['id'],row['arm']);assert key not in seen;seen.add(key)
        if key in ix:
            assert row['id']==m['control_id']
            for k in ('generated_ids','bounded_context_ids_sha256','position_sha256','source_state','pixel_sha256'):
                assert row.get(k)==ix[key].get(k),(key,k)
        else:
            assert row['id'] in m['missing_ids'];appended.append(row)
    assert len(appended)==68 and len(new)==72 and len(seen)==72
    merged=old+appended
    expected={(q['id'],a) for q in read(P/'questions.json') for a in ARMS}
    assert len(merged)==600 and {(x['id'],x['arm']) for x in merged}==expected
    out.mkdir()
    (out/'answers.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in merged),encoding='utf-8')
    save(out/'completed.json',dict(status='complete',answers=600,scope='merged_evaluation_not_one_successful_job',
        sources=[dict(run=m['parent'],status='88-minute timeout',sha256=sha(parent/'answers.jsonl'),answers_used=532),
                 dict(run=tail.name,sha256=sha(tail/'answers.jsonl'),answers_used=68,replays_excluded=4)],
        wall_costs_not_combined=True,model_loads=2,training=0))
    print('Merged 600 unique answers; parent failure retained; four replay answers excluded.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('tail',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();merge(a.tail,a.out)
