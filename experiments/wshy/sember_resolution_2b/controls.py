"""Three explicit input presets and a label-blind historical reuse schedule."""
import hashlib
from ..sember_input_diagnostic.controls import rotate

PRESETS = {'low': (192, 256), 'mid': (288, 384), 'high': (384, 512)}
ARMS = ('LS', 'A06', 'native', 'text_only')
HISTORICAL = {'B': ('native', 'low'), 'R': ('native', 'mid'), 'T': ('text_only', 'none')}


def geometry(name):
    h, w = PRESETS[name]
    p = (h // 32) * (w // 32)
    return dict(canvas=[h, w], grid=[h // 32, w // 32], pair_tokens=p,
                recent_tokens=4*p, history_tokens=4*p, total_tokens=8*p)


def canvas(frame, name):
    from PIL import Image
    h, w = PRESETS[name]
    # Preserve the historical two-stage letterbox at all three 3:4 canvases.
    content_h = h
    source = frame.to_image()
    source.thumbnail((w, w), Image.Resampling.BICUBIC)
    scale = min(w/source.width, content_h/source.height)
    size = (max(1, min(w, round(source.width*scale))),
            max(1, min(content_h, round(source.height*scale))))
    source = source.resize(size, Image.Resampling.BICUBIC)
    result = Image.new('RGB', (w, h))
    result.paste(source, ((w-size[0])//2, (h-size[1])//2))
    return result


def replay_ids(queries):
    """One SHA-ranked question per video, before consulting any answer."""
    groups = {}
    for q in queries:
        groups.setdefault(q['source_path'], []).append(q['id'])
    return sorted(min(ids, key=lambda qid: (hashlib.sha256(qid.encode()).hexdigest(), qid))
                  for ids in groups.values())


def expected_keys(queries):
    return {(q['id'], a, r, s) for q in queries for s in range(5)
            for a, r in [(a, r) for a in ARMS[:3] for r in PRESETS] + [('text_only', 'none')]}


def schedule(queries):
    if len(queries) != 102 or len({q['id'] for q in queries}) != 102:
        raise ValueError('all 102 unique development questions required')
    replay = replay_ids(queries)
    if len(replay) != 18:
        raise ValueError('all 18 source videos required')
    return dict(questions=102, videos=18, reused_answers=1530,
                new_native=510, new_memory=3060, replay_answers=270,
                maximum_new_answers=3840, maximum_new_tokens=30720,
                unique_analysis_answers=len(expected_keys(queries)), replay_ids=replay)


def validate_history(queries, predictions):
    index = {(r['id'], r['arm'], r['shift']): r for r in predictions}
    if len(index) != len(predictions) or set(index) != {
            (q['id'], a, s) for q in queries for a in ('B', 'R', 'F', 'T') for s in range(5)}:
        raise ValueError('historical 2040-answer grid incomplete or duplicated')
    result = []
    for q in queries:
        for old, (arm, resolution) in HISTORICAL.items():
            for shift in range(5):
                row = index[q['id'], old, shift]
                if (row['status'] != 'ok' or row['source_path'] != q['source_path']
                        or row['query_time_s'] != q['query_time_s']
                        or row['display_to_original'] != [(j+shift) % 5 for j in range(5)]
                        or not 1 <= len(row['generated_ids']) <= 8):
                    raise ValueError('historical identity or mapping mismatch')
                result.append(dict(row, arm=arm, resolution=resolution,
                                   historical_arm=old, provenance='job218446'))
    return result
