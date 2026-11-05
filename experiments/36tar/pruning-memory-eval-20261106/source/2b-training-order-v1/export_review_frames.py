"""Export bounded, PTS-labelled evidence sheets from the frozen review queue."""
import argparse
import hashlib
import json
from pathlib import Path

import av
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'runs/training_order_profile_20261104_01'
MEDIA = ROOT / 'runs/delayed_training_candidates_20261104_01'


def export(path, samples, target):
    wanted = {sample['frame_index']: (index, sample) for index, sample in enumerate(samples)}
    sheet = Image.new('RGB', (6 * 192, 8 * 160), 'white')
    draw = ImageDraw.Draw(sheet)
    found = 0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_count = 1
        for index, frame in enumerate(container.decode(stream)):
            if index not in wanted:
                continue
            slot, sample = wanted[index]
            image = ImageOps.contain(frame.to_image().convert('RGB'), (384, 288), Image.Resampling.BICUBIC)
            canvas = Image.new('RGB', (384, 288))
            canvas.paste(image, ((384 - image.width) // 2, (288 - image.height) // 2))
            if frame.pts != sample['raw_pts'] or hashlib.sha256(canvas.tobytes()).hexdigest() != sample['rgb_sha256']:
                raise ValueError(f'Sample mismatch: {path.name} at {index}')
            x, y = (slot % 6) * 192, (slot // 6) * 160
            sheet.paste(canvas.resize((192, 144), Image.Resampling.LANCZOS), (x, y))
            draw.text((x + 3, y + 145), f"{slot} | {sample['pts_s']:.2f}s", fill='black')
            found += 1
            if found == len(samples):
                break
    if found != len(samples):
        raise ValueError(f'Missing samples: {path.name}')
    sheet.save(target, quality=90)
    return hashlib.sha256(target.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-offset', type=int, default=0)
    parser.add_argument('--train-count', type=int, required=True)
    parser.add_argument('--include-development', action='store_true')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if not args.out.resolve().is_relative_to(RUN.resolve()):
        raise ValueError('Output must remain in the review run')
    if args.train_offset < 0 or not 0 <= args.train_count <= 16:
        raise ValueError('Invalid review batch bounds')
    queue = json.loads((RUN/'next_evidence_review_LOCAL_ONLY.json').read_text(encoding='utf-8'))['sources']
    selected = queue['train_candidate'][args.train_offset:args.train_offset + args.train_count]
    if args.include_development:
        selected += queue['development_candidate']
    if not selected:
        raise ValueError('Empty review batch')
    audit = {row['video_id']: row for row in map(json.loads, (MEDIA/'media_audit.jsonl').read_text().splitlines())}
    plan = json.loads((MEDIA/'acquisition_plan.json').read_text())
    sources = {row['video_id']: row for row in plan['files']}
    args.out.mkdir()
    records = []
    for row in selected:
        video = row['video_id']
        samples = audit[video]['samples'][:48]
        assert len(samples) == 48
        path = ROOT/'datasets/delayed-training-nextqa-v1'/sources[video]['mirror_path']
        target = args.out/f'{video}.jpg'
        records.append(dict(row, sheet=target.name, sha256=export(path, samples, target), sampled_frames=len(samples)))
    (args.out/'manifest_LOCAL_ONLY.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    print(json.dumps(dict(sources=len(records), sampled_frames=48*len(records))))


if __name__ == '__main__':
    main()
