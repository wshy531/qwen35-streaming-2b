"""Integrity, legacy replay and preregistered quality/QA-cost gate."""
import argparse,json
from pathlib import Path
from common import ROOT,OUT,ARMS,POLICIES,read,save,sha
from analyze import analyze

def main(result):
    done=read(result/'completed.json');assert done['status']=='complete' and done['answers']==1818 and done['vision_encodings']==3981
    rows=[json.loads(l) for l in (result/'answers.jsonl').read_text(encoding='utf-8').splitlines()]
    tokens=[json.loads(l) for l in (result/'tokens.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rows)==len({(r['id'],r['arm']) for r in rows})==1818
    keyed={(r['id'],r['arm']):r for r in rows};token_map={}
    for t in tokens:token_map.setdefault((t['query_id'],t['arm']),[]).append(t)
    ids={q['id'] for q in read(OUT/'inputs/questions.json')}
    for uid in ids:
        for p in POLICIES:
            full=keyed[uid,p+'_full48'];half=keyed[uid,p+'_read24']
            assert full['storage_node_digest']==half['storage_node_digest']
            assert full['visual_tokens']==half['visual_tokens']*2
            assert [(n['first'],n['last'],n['begin'],n['end']) for n in full['nodes']]==[(n['first'],n['last'],n['begin'],n['end']) for n in half['nodes']]
        for arm in ARMS:
            r=keyed[uid,arm];ts=token_map[uid,arm]
            assert r['banks_unchanged'] and r['branch_released'] and r['generated_ids']==[t['token_id'] for t in ts]
            assert [t['token_index'] for t in ts]==list(range(len(ts))) and r['first_token_seconds']>=0
    # replay() itself checks the exact frozen per-query submitted prefix.
    assert all(r['committed_frames_sha256'] for r in rows if r['cohort']=='stream34')
    assert len(done['profiles'])==112 and all(p['source']['completed'] and p['source']['codec']['requested_threads']==4 for p in done['profiles'])
    analyze(result,OUT/'answer_results.json');scores=read(OUT/'answer_results.json')
    replay=scores['historical_replay'];assert replay['expected']==909
    assert not any(replay[k] for k in ('generated_ID_mismatches','input_ID_mismatches','node_mismatches'))
    gates={};tables=scores['tables']
    for p in POLICIES:
        f=p+'_full48';h=p+'_read24'
        cost={a:sum(r['QA_total_seconds'] for r in rows if r['arm']==a) for a in (f,h)}
        gain=1-cost[h]/cost[f]
        quality=all(tables[s]['scores'][h]['correct']>=tables[s]['scores'][f]['correct'] for s in ('primary269','stream_supported16'))
        gates[p]=dict(primary269_full=tables['primary269']['scores'][f]['correct'],primary269_read=tables['primary269']['scores'][h]['correct'],
            long16_full=tables['stream_supported16']['scores'][f]['correct'],long16_read=tables['stream_supported16']['scores'][h]['correct'],
            QA_seconds=cost,QA_reduction=gain,quality_gate=quality,engineering_gate=gain>=.15,eligible=quality and gain>=.15)
    chosen=next((p for p in ('LS','A06','M2') if gates[p]['eligible']),None)
    save(OUT/'integrity_and_decision.json',dict(status='passed',answers=1818,tokens=len(tokens),legacy_replay=909,
        main_banks_unchanged=True,matched_node_coverage=True,gates=gates,chosen_for_new_source_confirmation=chosen,
        default_enabled=False,new_source_validation_performed=False,
        conclusion='candidate passes development gate; pending independent data' if chosen else 'close this first version; no parameter sweep',
        limitations=['No net loss is not per-question losslessness or statistical noninferiority',
            'Shared visual execution does not establish independent full-pipeline speed',
            'Storage stays384 perbank; only temporary readout reduced',
            'P3 write-time scores are context, not pure placement causal comparison']))
    print(dict(status='passed',gates=gates,chosen=chosen))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);main(p.parse_args().results)
