"""CPU-only; preserves failed gates and never reports partial QA as full evaluation."""
import argparse,json,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
def rows(name):
    f=a.run/name
    return [json.loads(s) for s in f.read_text().splitlines()] if f.exists() else []
alignment=rows('alignment.jsonl');answers=rows('answers.jsonl');probes=rows('train_probes.jsonl');visual=rows('visual.jsonl')
completed=json.loads((a.run/'completed.json').read_text()) if (a.run/'completed.json').exists() else {}
out={'status':completed.get('status','partial_or_failed'),
     'alignment_count':len(alignment),'answer_count':len(answers),'train_probe_count':len(probes)}
if alignment:out['alignment']={'max_abs':max(r['max_abs'] for r in alignment),'max_relative_l2':max(r['relative_l2'] for r in alignment),'argmax_mismatches':sum(not r['argmax_equal'] for r in alignment)}
if (a.run/'failed.json').exists():out['failure']=json.loads((a.run/'failed.json').read_text())
if out['status']=='complete':
    assert len(answers)==2070 and len({(r['id'],r['arm']) for r in answers})==2070
    out['quality']={}
    for arm in sorted({r['arm'] for r in answers}):
        z=[r for r in answers if r['arm']==arm];assert len(z)==345
        out['quality'][arm]={'correct':sum(r['correct'] for r in z),'questions':len(z)}
    out['paired']={}
    index={(r['id'],r['arm']):r for r in answers};ids={r['id'] for r in answers}
    for left,right in [('final_none','initial_none'),('final_original','final_none'),('initial_final_bias','initial_none'),('final_initial_bias','initial_original'),('final_original','final_initial_bias')]:
        gain=sum(index[q,left]['correct'] and not index[q,right]['correct'] for q in ids)
        loss=sum(not index[q,left]['correct'] and index[q,right]['correct'] for q in ids)
        out['paired'][left+' vs '+right]={'gain':gain,'loss':loss,'net':gain-loss}
if probes:
    out['train_CE']={arm:statistics.mean(r['full_CE'] for r in probes if r['arm']==arm) for arm in {r['arm'] for r in probes}}
if visual:
    v={(r['video_id'],r['pair']):r for r in visual};z=list(v.values())
    out['visual']={'unique_pairs':len(z),'mean_overlap_out_of54':statistics.mean(r['selection_overlap'] for r in z),
       'bias':{n:{'min':min(r[n]['bias_min'] for r in z),'max_span':max(r[n]['bias_span'] for r in z),'mean_span':statistics.mean(r[n]['bias_span'] for r in z),'below_minus2':sum(r[n]['below_minus2'] for r in z)} for n in ['initial','final']}}
(a.run/'diagnostic_analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
