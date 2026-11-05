"""Report all same-case readouts and selection changes, without accuracy claims."""
import argparse,json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);args=p.parse_args()
    done=json.loads((args.folder/'completed.json').read_text())
    probes=json.loads((args.folder/'probes.json').read_text())
    curves={a:[json.loads(s) for s in (args.folder/(a+'_updates.jsonl')).read_text().splitlines()]
            for a in ('whole_first','whole_repeat','recompute')}
    assert all(len(rows)==4 for rows in curves.values()) and done['updates']==12
    changes=[]
    for i,initial in enumerate(probes['initial']):
        for arm in curves:
            current=probes[arm][i];ref=probes['whole_first'][i]
            assert current['id']==initial['id']==ref['id']
            for readout in ('LS','recent'):
                x=current['readouts'][readout];base=initial['readouts'][readout]
                changes.append(dict(id=current['id'],arm=arm,readout=readout,
                    ce_initial=base['ce'],ce_final=x['ce'],ce_change=x['ce']-base['ce'],
                    option_initial=base['option_argmax'],option_final=x['option_argmax']))
    selected={}
    for arm in curves:
        changed,total,dropped=0,0,0
        for observed,reference in zip(probes[arm],probes['whole_first']):
            for pair,ids in observed['selected_ids'].items():
                ref=reference['selected_ids'][pair];total+=1;changed+=ids!=ref
                dropped+=len(set(ref)-set(ids))
        selected[arm]=dict(changed_pairs=changed,total_pairs=total,different_retained_groups=dropped)
    result=dict(completion=done,training_ce={a:[r['ce'] for r in rows] for a,rows in curves.items()},
                same_case_probe_changes=changes,selection_differences_vs_whole=selected,
                limits='Same two training cases; selected option argmax is not generated-answer accuracy.')
    with (args.folder/'analysis.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps(dict(status=done['status'],cpu_tiny=done['cpu_tiny'],
                         functional_screen_passed=done['functional_screen_passed'],selections=selected)))


if __name__=='__main__':main()
