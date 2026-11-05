"""Materialize all deterministic media exclusions, without consulting model answers."""
import collections
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/learned_prune_data_prepare_20261101_01'


def main():
    audit_path = OUT / 'CPU_media_audit.json'
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    assert audit['status'] == 'admission_gate_passed', 'stop: insufficient eligible media'
    sources = json.loads((OUT / 'sources.json').read_text(encoding='utf-8'))
    questions = json.loads((OUT / 'questions.json').read_text(encoding='utf-8'))
    admitted = {r['video_id']: r for r in audit['rows'] if r['admitted']}
    qs = [q for q in questions if q['video_id'] in admitted]
    ss = [dict(s, media_verified=True, duration_s=admitted[s['video_id']]['duration_s'],
               real_frames=admitted[s['video_id']]['real_frames'],
               encoding_pairs=admitted[s['video_id']]['encoding_pairs'])
          for s in sources if s['video_id'] in admitted]
    summary = dict(status='media_admitted_not_answer_support_certified', videos=len(ss), questions=len(qs),
                   roles={role: dict(videos=sum(s['role']==role for s in ss),
                                     questions=sum(q['role']==role for q in qs),
                                     types=dict(collections.Counter(q['type'] for q in qs if q['role']==role)))
                          for role in ['train','development','confirmation']},
                   duration_range_s=[min(s['duration_s'] for s in ss),max(s['duration_s'] for s in ss)],
                   duration_total_s=sum(s['duration_s'] for s in ss),
                   duration_bins={label:sum(lo <= s['duration_s'] < hi for s in ss)
                                  for label,lo,hi in [('0-10s',0,10),('10-15s',10,15),('15-20s',15,20.0001)]},
                   real_frames_total=sum(s['real_frames'] for s in ss),
                   encoding_pairs_total=sum(s['encoding_pairs'] for s in ss),
                   excluded=[dict(video_id=r['video_id'], reasons=r['reasons']) for r in audit['rows'] if not r['admitted']],
                   audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),
                   source_parent_independence='unknown', training_updates=0)
    for name,data in [('admitted_sources.json',ss),('admitted_questions.json',qs),('admission_summary.json',summary)]:
        (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == '__main__': main()
