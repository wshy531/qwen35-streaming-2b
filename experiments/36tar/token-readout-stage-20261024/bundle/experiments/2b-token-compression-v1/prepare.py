"""Freeze 269 opened questions plus a separate34 causal-stream regression; no pixels/GPU."""
from collections import defaultdict
from pathlib import Path
import sys

from common import ROOT, OUT, read, save, sha, ARMS


def main():
    OUT.mkdir(exist_ok=False)
    run = OUT / 'inputs'
    run.mkdir()
    cat = ROOT / 'runs/dataset_catalog_20261021_01/catalog'
    media = {a['sha256']: a for a in read(cat / 'media.json')}
    groups, questions, labels = [], [], []
    provenance = {}
    sources = [
        ('natural171', ROOT / 'runs/2b_natural_eval_20261020_01/requests.json',
         ROOT / 'runs/2b_natural_eval_20261020_01/scorer_rows_local.json'),
        ('T98', ROOT / 'runs/2b_validation_T_prepare_20261020_01/runtime_inputs/requests.json',
         ROOT / 'runs/2b_validation_T_prepare_20261020_01/scorer_rows_LOCAL_ONLY.json')]
    for cohort, req, scorer in sources:
        rows = read(req)
        truth = {r['id']: r for r in read(scorer)}
        provenance[req.relative_to(ROOT).as_posix()] = sha(req)
        provenance[scorer.relative_to(ROOT).as_posix()] = sha(scorer)
        by_media = defaultdict(list)
        for row in rows:
            by_media[row['media_sha256']].append(row)
        for h, items in sorted(by_media.items()):
            a = media[h]
            frames = items[0]['model_visible']['frames']
            assert all(r['model_visible']['frames'] == frames for r in items)
            gid = cohort + ':' + h[:16]
            groups.append(dict(id=gid, cohort=cohort, sha256=h, local_media=a['local_path'],
                cluster_media=a['cluster_path'], frames=frames, duration_s=a['duration_s'],
                query_policy='existing_full_clip_query_snapshot', odd_tail='duplicate_last_only_at_query_snapshot',
                question_ids=[r['id'] for r in items]))
            for row in items:
                v = row['model_visible']
                assert all(f['pts_s'] <= v['query_time_s'] + 1e-9 for f in frames)
                q = truth[row['id']]
                original = q.get('original', q)
                assert v['question'] == original['question'] and v['options'] == original['options']
                questions.append(dict(id=row['id'],group=gid,cohort=cohort,question=v['question'],
                    options=v['options'],query_time_s=v['query_time_s'],pairs=(len(frames)+1)//2))
                labels.append(dict(id=row['id'],cohort=cohort,answer_index=original['answer_index'],
                    stratum=q['stratum'] if cohort=='T98' else q['curation']['stratum']))
    long = ROOT / 'runs/2b_long_stream_prepare_20261020_01/runtime_inputs_v2'
    labels34 = read(long.parent / 'stream_labels34_LOCAL_ONLY.json')
    longq = read(long / 'requests.json')
    snaps = {s['query_id']: s for s in read(long / 'snapshots.json')}
    for p in [long/'sources.json',long/'requests.json',long/'snapshots.json',long.parent/'stream_labels34_LOCAL_ONLY.json']:
        provenance[p.relative_to(ROOT).as_posix()] = sha(p)
    for source in read(long / 'sources.json'):
        h = source['media_sha256']
        a = media[h]
        items = [q for q in longq if q['source_path'] == source['source_path']]
        gid = 'stream34:' + h[:16]
        groups.append(dict(id=gid,cohort='stream34',sha256=h,local_media=a['local_path'],
            cluster_media=a['cluster_path'],frames=source['frames'],duration_s=source['duration_s'],
            source_path=source['source_path'],query_policy='existing_causal_midstream_complete_pairs',
            odd_tail='never_commit_pending_frame_at_query',question_ids=[q['id'] for q in items]))
        for q in items:
            snapshot = snaps[q['id']]
            questions.append(dict(id=q['id'],group=gid,cohort='stream34',question=q['question'],
                options=q['options'],query_time_s=q['query_time_s'],pairs=snapshot['committed_pairs'],
                committed_frames_sha256=snapshot['committed_frames_sha256']))
    labels += [dict(**row,cohort='stream34') for row in labels34]
    assert len(questions) == len({q['id'] for q in questions}) == 303
    assert len(groups) == 112 and len(labels) == 303
    assert sum(q['cohort']=='natural171' for q in questions) == 171
    assert sum(q['cohort']=='T98' for q in questions) == 98
    assert max((len(g['frames'])+1)//2 for g in groups) <= 320
    save(run/'groups.json',groups)
    save(run/'questions.json',questions)
    save(OUT/'labels_LOCAL_ONLY.json',labels)
    save(OUT/'input_provenance.json',provenance)
    pairs=sum(len(g['frames'])//2 if g['cohort']=='stream34' else (len(g['frames'])+1)//2 for g in groups)
    save(run/'protocol.json',dict(status='CPU_preparation_no_GPU_authorization',arms=list(ARMS),
        questions=303,primary_open_development=269,separate_midstream_regression=34,
        groups=112,answers_max=909,generated_tokens_max=7272,vision_encodings=pairs,
        new_candidate='none: existing odd selection and pair-local ToMe against unchanged full features',
        hypothesis='At equal24 coordinates and token budget, do merged values introduce more reading loss than unchanged selected values?',
        model_revision='15852e8c16360a2fea060d615a32b45270f8a8fc',dtype='FP32',
        canvas_hw=[192,256],question_suffix_cap=128,max_new_tokens=8,context_cap=20000,
        pair_archive_cap=320,feature_archive_bytes_cap_at2048=252825600,
        archive='full+two half versions of one finite clip, explicitly extra diagnostic memory; NOT bounded long-term algorithm',
        equal_time_labels=True,equal_odd_tome_positions=True,no_cross_time_merge=True,
        main_tokens=2048,main_model_state_not_compressed=True,
        causal_question_blind_writes=True,wallclock_1x=False,
        nominal_scope='single fixed comparison; no ratio/threshold/prompt sweep, no training',
        stop='score all303 and report each cohort/stratum separately; no added variants; no default upgrade or generalization claim',
        old_S24_not_rerun='S24 mixed pair compression with subsequent temporal forest; this isolates a single pair reduction with all times retained',
        no_claims=['no ViT acceleration from projected reduction','no unbounded memory','no end-to-end speedup from token count alone'],
        input_sha256={p.name:sha(p) for p in run.iterdir()}))
    print(dict(groups=len(groups),questions=len(questions),answers=909,visual_pairs=pairs,GPU_authorized=False))


if __name__=='__main__':
    main()
