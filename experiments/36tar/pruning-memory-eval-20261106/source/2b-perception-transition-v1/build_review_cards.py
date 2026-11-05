"""Reassemble exact sampled RGB tiles; never downloads or alters video."""
import hashlib
from PIL import Image, ImageDraw
from transition_common import ROOT, PREP, INPUTS, SOURCE, read, save, sha


def main():
    index = {}
    for receipt_path in sorted((SOURCE / 'details_LOCAL_ONLY').glob('*/receipt.json')):
        receipt = read(receipt_path)
        if not receipt.get('native_input_canvas'):
            continue
        for page in receipt['pages']:
            if not page.get('input_RGB_exact'):
                continue
            image_path = receipt_path.parent / page['path'].replace('\\', '/').split('/')[-1]
            assert sha(image_path) == page['sha256']
            for i, frame in enumerate(page['frames']):
                key = receipt['source_sha256'], frame['frame_id']
                value = dict(image_path=image_path, image_sha256=page['sha256'], tile_index=i,
                    top=24, columns=4, stride=312, frame=frame)
                if key in index:
                    assert index[key]['frame']['mid_rgb_sha256'] == frame['mid_rgb_sha256']
                else:
                    index[key] = value
    extra = read(PREP / 'missing_native_frames_receipt.json')
    extra_path = PREP / extra['path']
    assert sha(extra_path) == extra['sha256']
    for i, frame in enumerate(extra['frames']):
        index[(extra['source_sha256'], frame['frame_id'])] = dict(image_path=extra_path,
            image_sha256=extra['sha256'], tile_index=i, top=extra['top'], columns=4, stride=312, frame=frame)
    sources = {r['name']:r for r in read(INPUTS / 'sources_LOCAL_ONLY.json')}
    cards = []
    for task in read(INPUTS / 'tasks_LOCAL_ONLY.json'):
        if task['role'] != 'current':
            continue
        result = Image.new('RGB', (1536, 624), 'white')
        draw = ImageDraw.Draw(result)
        opened = {}
        origins = []
        for i, frame in enumerate(task['recent_frames']):
            source = index[(sources[task['source_name']]['media_sha256'], frame['frame_id'])]
            image_path = source['image_path']
            if image_path not in opened:
                opened[image_path] = Image.open(image_path).convert('RGB')
            tile = source['tile_index']
            x, y = tile % 4 * 384, source['top'] + tile // 4 * source['stride']
            rgb = opened[image_path].crop((x, y, x+384, y+288))
            assert hashlib.sha256(rgb.tobytes()).hexdigest() == frame['mid_rgb_sha256']
            target_x, target_y = i % 4 * 384, i // 4 * 312
            draw.text((target_x+3,target_y+3), f"{frame['pts_s']:.3f}s f{frame['frame_id']}", fill='black')
            result.paste(rgb, (target_x,target_y+24))
            origins.append(dict(frame=frame, image_relative_path=image_path.relative_to(ROOT).as_posix(),
                source_image_sha256=source['image_sha256'], source_tile_index=tile))
        path = PREP / 'review_cards_LOCAL_ONLY' / (task['id'].replace(':','_') + '.png')
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(path)
        result.save(path)
        cards.append(dict(task_id=task['id'], relative_path=path.relative_to(ROOT).as_posix(),
            sha256=sha(path), bytes=path.stat().st_size, input_RGB_exact=True,
            last_pair_frames=task['evidence_frames'], all_recent_frames=task['recent_frames'], origins=origins,
            top=24, columns=4, row_stride=312, native_tile_hw=[288,384],
            assistant_visual_review='pending', independent_review=False))
        for image in opened.values():
            image.close()
    assert len(cards) == 12
    save(PREP / 'review_card_receipt_LOCAL_ONLY.json', dict(status='exact_recent_RGB_cards_assembled',
        cards=cards, total_bytes=sum(c['bytes'] for c in cards), raw_frames_verified=96,
        videos_downloaded=0, model_calls=0, GPU_jobs=0))
    print(dict(cards=12, exact_sampled_RGB_tiles=96, bytes=sum(c['bytes'] for c in cards), GPU_jobs=0))


if __name__ == '__main__':
    main()
