"""Sample timestamps around two official anchors for a local visual review."""
import hashlib

import av
from PIL import Image, ImageDraw

from fetch_validation import ROOT, OUT, read, save


def sample(path, targets):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_count = 1
        start = float((stream.start_time or 0) * stream.time_base)
        container.seek(int((start + max(0, targets[0] - 1)) / stream.time_base),
                       stream=stream, backward=True)
        pos = 0
        previous = None
        output = []
        for frame in container.decode(stream):
            if frame.time is None:
                continue
            pts = frame.time - start
            while pos < len(targets) and pts > targets[pos]:
                assert previous is not None
                image = previous[1].to_image()
                image.thumbnail((400, 224))
                output.append((previous[0], image))
                pos += 1
            if pos == len(targets):
                break
            previous = pts, frame
        assert len(output) == len(targets)
        return output


def main():
    queue = read(OUT / 'diagnosis_evidence_review_queue.json')
    destination = OUT / 'diagnosis_contact_sheets'
    destination.mkdir(exist_ok=False)
    manifest = []
    for row in queue:
        path = ROOT / row['media_path']
        sheet = Image.new('RGB', (1200, 1008), 'white')
        draw = ImageDraw.Draw(sheet)
        records = {}
        for i, anchor in enumerate(('early_anchor', 'current_anchor')):
            query = row[anchor]['query_time_s']
            targets = [max(0, query - offset) for offset in (12, 8, 4, 2, 1, 0)]
            frames = sample(path, targets)
            records[anchor] = dict(query_time_s=query, frame_pts=[p for p, _ in frames],
                                   question_id=row[anchor]['id'])
            for j, (pts, frame) in enumerate(frames):
                x, y = j % 3 * 400, (i * 2 + j // 3) * 252
                sheet.paste(frame, (x + (400 - frame.width) // 2, y))
                draw.text((x + 12, y + 229), f'{anchor}: {pts:.2f}s / query {query:.1f}s', fill='black')
        target = destination / f'{path.stem}.jpg'
        sheet.save(target, quality=88)
        manifest.append(dict(source_id=row['source_id'], sheet=target.relative_to(ROOT).as_posix(),
                             sheet_sha256=hashlib.sha256(target.read_bytes()).hexdigest(), anchors=records))
    save(destination / 'manifest.json', manifest)
    print(f'{len(manifest)} sources prepared for visual review')


if __name__ == '__main__':
    main()
