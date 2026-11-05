"""Bounded source-overlap screen without viewing holdout images or labels.

Exact asset/sample hashes and five fixed perceptual samples per source are
screens, not proof of filming-session independence. Low-texture/title frames
are reported separately; two different matching pairs are required to flag a
perceptual source pair. No candidate is removed or re-split automatically.
"""
from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path
import time

import av
from audit_sampling import canvas_bytes
from build_dataset_release import load_media_traces
from index_evaluation_media import fingerprint


def main():
    root=Path(__file__).resolve().parents[2];run=root/'runs/dataset_build_20261018_01'
    media,traces=load_media_traces()
    registry=json.loads((run/'source_registry.json').read_text(encoding='utf-8'))
    identity={s['source_path']:s for s in registry['sources']}
    plan=[]
    for source,t in traces.items():
        frames=t['frames'];indices=sorted({round((len(frames)-1)*i/4) for i in range(5)})
        plan.append(dict(source_path=source,frames=[frames[i] for i in indices]))
    (run/'overlap_screen_plan.json').write_text(json.dumps(dict(
        purpose='fixed_CPU_fingerprint_screen_no_holdout_semantic_review',
        selected_before_computing_scores=True,fractions=[0,.25,.5,.75,1],
        perceptual_max_hamming=4,minimum_gray_std=8,minimum_distinct_matching_pairs=2,
        sources=plan),indent=2)+'\n',encoding='utf-8')
    start=time.monotonic();sketches=[]
    for item in plan:
        selected=item['frames'];missing={f['frame_id']:f for f in selected if 'dhash64' not in f}
        derived={}
        if missing:
            video_start=time.monotonic()
            with av.open(media[item['source_path']]['local_path']) as c:
                stream=c.streams.video[0];stream.codec_context.thread_count=1
                for frame_id,frame in enumerate(c.decode(stream)):
                    if time.monotonic()-video_start>120:raise TimeoutError('per-source fingerprint budget')
                    if frame_id not in missing:continue
                    raw=canvas_bytes(frame)
                    import hashlib
                    if hashlib.sha256(raw).hexdigest()!=missing[frame_id]['rgb_sha256']:
                        raise ValueError('fingerprint differs from recorded actual sample')
                    derived[frame_id]=fingerprint(raw)
                    if len(derived)==len(missing):break
            if len(derived)!=len(missing):raise ValueError('selected sample not decoded')
        sketches.append(dict(source_path=item['source_path'],split=identity[item['source_path']]['split'],
            frames=[dict(f,**derived.get(f['frame_id'],{})) for f in selected]))
    exact=defaultdict(list);frame_hashes=defaultdict(list)
    for source,m in media.items():exact[m['sha256']].append(source)
    for source,t in traces.items():
        for f in t['frames']:frame_hashes[f['rgb_sha256']].append(dict(source_path=source,
            split=identity[source]['split'],frame_id=f['frame_id'],pts_s=f['pts_s']))
    repeated_frames=[dict(rgb_sha256=h,occurrences=v) for h,v in frame_hashes.items()
        if len({x['source_path'] for x in v})>1 and len({x['split'] for x in v})>1]
    flags=[]
    for a,b in combinations(sketches,2):
        if a['split']==b['split']:continue
        matches=[]
        for x in a['frames']:
            for y in b['frames']:
                if x['gray_std']<8 or y['gray_std']<8:continue
                distance=(int(x['dhash64'],16)^int(y['dhash64'],16)).bit_count()
                if distance<=4:matches.append(dict(a_frame=x['frame_id'],b_frame=y['frame_id'],distance=distance))
        if len({m['a_frame'] for m in matches})>=2 and len({m['b_frame'] for m in matches})>=2:
            flags.append(dict(a_source=a['source_path'],b_source=b['source_path'],
                a_split=a['split'],b_split=b['split'],matches=matches,
                status='weak_perceptual_candidate_not_confirmed_overlap'))
    report=dict(status='CPU_overlap_screen_not_independence_certificate',
        media_files=len(media),exact_duplicate_file_groups=[v for v in exact.values() if len(v)>1],
        cross_split_exact_sample_groups=repeated_frames,
        cross_split_perceptual_candidates=flags,sketches=sketches,elapsed_s=time.monotonic()-start,
        holdout_images_visually_reviewed=False,questions_or_GT_used=False,
        limits=['Byte equality can detect duplicates but different encodes/crops can evade it.',
            'Shared black/title frames and similar layouts are not proof of a shared recording.',
            'Five samples can miss overlap elsewhere; no flag is not proof of no leakage.',
            '10 unresolved Ego parents still block full parent-identity certification.',
            'No automatic candidate deletion or split changes.'])
    (run/'source_overlap_screen.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(media_files=len(media),exact_duplicate_file_groups=len(report['exact_duplicate_file_groups']),
        cross_split_exact_sample_groups=len(repeated_frames),cross_split_perceptual_candidates=len(flags),
        elapsed_s=report['elapsed_s'])))


if __name__=='__main__':main()
