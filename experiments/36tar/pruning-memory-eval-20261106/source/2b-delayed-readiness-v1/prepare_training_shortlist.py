"""Metadata-only, outcome-independent shortlist; NOT admitted delayed QA labels."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/delayed_training_candidates_20261104_01'
SEED = 'delayed-training-shortlist-20261104-v1'


def main():
    inventory = json.loads((ROOT / 'runs/delayed_profile_prepare_20261104_01/training_inventory.json').read_text())
    grouped = defaultdict(list)
    with (ROOT / 'runs/learned_prune_data_prepare_20261101_01/upstream/train.csv').open(
            encoding='utf-8', newline='') as handle:
        for row in csv.DictReader(handle):
            grouped[row['video_id']].append(row)
    bounds = [(481,900,8),(901,1500,7),(1501,2400,8),(2401,3600,7)]
    sources, questions = [], []
    for low,high,dev_count in bounds:
        # Require at least one official temporal question, not a guessed evidence interval.
        eligible = [v for v in inventory['candidates'] if low <= v['frame_count'] <= high
                    and any(t.startswith('T') for t in v['question_types'])]
        eligible.sort(key=lambda v:hashlib.sha256((SEED+v['video_id']).encode()).hexdigest())
        assert len(eligible) >= 30+dev_count
        for i,source in enumerate(eligible[:30+dev_count]):
            role = 'train_candidate' if i < 30 else 'development_candidate'
            sources.append(dict(source,role=role,frame_count_stratum=[low,high],
                                duration_s=None,parent_capture_id=None,media_acquired=False))
            for q in grouped[source['video_id']]:
                options = [q['a'+str(j)] for j in range(5)]
                matches = [j for j,a in enumerate(options) if a == q['answer']]
                fingerprint=hashlib.sha256(json.dumps(q,sort_keys=True).encode()).hexdigest()[:12]
                questions.append(dict(id=f"nextqa-train-{q['video_id']}-{q['qid']}-{fingerprint}",
                    video_id=q['video_id'],official_qid=q['qid'],role=role,official_type=q['type'],
                    question=q['question'],options=options,answer_text=q['answer'],
                    answer_index=matches[0] if len(matches)==1 else None,
                    original_split='NExT-QA train',temporal_candidate=q['type'].startswith('T'),
                    status='pending_media_and_evidence_review',evidence_intervals=[],
                    query_cutoff_s=None,later_clues=None))
    assert len(sources)==150 and len({s['video_id'] for s in sources})==150
    prior = json.loads((ROOT/'runs/learned_prune_data_prepare_20261101_01/sources.json').read_text())
    assert not {s['video_id'] for s in sources} & {s['video_id'] for s in prior}
    assert len({q['id'] for q in questions})==len(questions)
    OUT.mkdir()
    summary = dict(status='metadata_shortlist_not_admitted_training_data',seed=SEED,
        videos=len(sources),roles=dict(Counter(s['role'] for s in sources)),
        official_questions=len(questions),question_roles=dict(Counter(q['role'] for q in questions)),
        question_types=dict(Counter(q['official_type'] for q in questions)),
        temporal_questions=sum(q['temporal_candidate'] for q in questions),
        bytes=sum(s['bytes'] for s in sources),verified_delayed_questions=0,
        ambiguous_answer_mapping=sum(q['answer_index'] is None for q in questions),
        repeated_official_video_qid_keys=sum(count>1 for count in
            Counter((q['video_id'],q['official_qid']) for q in questions).values()),
        downloaded=0,excluded_previous_files=len(prior),
        duration_or_parent_separation_verified=False,
        limitation='Frame count strata are not seconds. No model scores used. No future frames permitted at query.')
    for filename,value in [('sources.json',sources),('questions_LOCAL_ONLY.json',questions),('summary.json',summary)]:
        (OUT/filename).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))


if __name__ == '__main__':
    main()
