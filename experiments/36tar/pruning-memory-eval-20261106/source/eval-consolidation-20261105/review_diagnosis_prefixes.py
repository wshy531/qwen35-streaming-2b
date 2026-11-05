"""Export the exact sampled prefixes used to review four diagnosis candidates."""
from collections import deque
import hashlib
import sys

from PIL import Image, ImageDraw

from fetch_validation import ROOT, OUT, read, save

sys.path.insert(0, str(ROOT / 'experiments/streamingbench-router-v1'))
from v0_source import V0Pairs

REVIEW = OUT / 'diagnosis_prefix_review'
# Visually located evidence windows from the earlier anchor review.
EARLY_WINDOWS = {'sample_205_real.mp4': (0, 4), 'sample_202_real.mp4': (60, 72),
                 'sample_156_real.mp4': (93, 98), 'sample_20_real.mp4': (0, 12)}


def sheet(path, packets):
    columns = 4
    height = ((len(packets) + columns - 1) // columns) * 312
    image = Image.new('RGB', (384 * columns, height), 'white')
    draw = ImageDraw.Draw(image)
    for n, packet in enumerate(packets):
        x, y = n % columns * 384, n // columns * 312
        image.paste(Image.fromarray(packet.rgb), (x, y))
        draw.text((x + 8, y + 293), f'frame {packet.frame_id}  PTS {packet.timestamp:.3f}s', fill='black')
    image.save(path, quality=92)
    return dict(path=path.relative_to(ROOT).as_posix(), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                timestamps=[p.timestamp for p in packets],
                rgb_sha256=[hashlib.sha256(p.rgb.tobytes()).hexdigest() for p in packets])


def main():
    REVIEW.mkdir(exist_ok=False)
    rows = []
    for source in read(OUT / 'diagnosis_evidence_review_queue.json'):
        name = source['source_id']
        early_start, early_end = EARLY_WINDOWS[name]
        cutoff = source['current_anchor']['query_time_s']
        recent_pairs = deque(maxlen=8)
        early_packets = []

        def commit(first, second, padding=False):
            assert not padding and second.timestamp <= cutoff
            recent_pairs.append((first, second))
            early_packets.extend(p for p in (first, second) if early_start <= p.timestamp <= early_end + .05)

        sampler = V0Pairs(ROOT / source['media_path'], commit, config='auto4')
        try:
            status = sampler.advance(cutoff)
        finally:
            sampler.close()
        recent = [p for pair in recent_pairs for p in pair]
        assert len(recent) == 16 and early_packets and early_packets[-1].timestamp < recent[0].timestamp
        rows.append(dict(source_id=name, cutoff_s=cutoff, sampler_status=status,
                         early=sheet(REVIEW / f'{name[:-4]}_early.jpg', early_packets),
                         recent=sheet(REVIEW / f'{name[:-4]}_recent8.jpg', recent)))
    save(REVIEW / 'frames.json', rows)
    print(f'{len(rows)} reviewed prefixes extracted; {sum(r["sampler_status"]["complete_pairs"] for r in rows)} sampled pairs')


if __name__ == '__main__':
    main()
