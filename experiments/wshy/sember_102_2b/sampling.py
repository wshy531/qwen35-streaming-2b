"""Generate media identities using the frozen 1 FPS canvas sampler."""
import os

from ..online_memory_2b.isolation import configure, write_root
configure()
from ..online_memory_2b.frozen import NATURAL, module
from ..online_memory_2b.protocol import read, safe_path, save, sha
from .inputs import base_inputs, bundle_path, snapshots


def media():
    if (not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES') != ''
            or os.environ.get('SLURM_CPUS_PER_TASK') != '4'):
        raise ValueError('media preparation requires approved allocation with GPU hidden')
    _, groups, files = base_inputs()
    receipt = read(bundle_path() / 'relay-20261024T055516144935Z/receipt.json')
    downloaded = {entry['video_id']: entry for entry in receipt}
    if len(receipt) != 18 or len(downloaded) != 18:
        raise ValueError('complete prior download receipt required')
    out = bundle_path() / 'sampling'
    out.mkdir(exist_ok=False)
    (out / 'frames').mkdir()
    sampler = module('audit_sampling', NATURAL)
    sources, prefixes = [], {}
    for entry in files:
        relative = 'data/S-EMBER/' + entry['hf_revision'] + '/' + entry['relative_path']
        path = safe_path(write_root(), relative)
        if path.stat().st_size != entry['expected_bytes']:
            raise ValueError('source size mismatch')
        before = sha(path)
        identity = downloaded[entry['video_id']]
        if (before != identity['sha256'] or identity['bytes'] != entry['expected_bytes']
                or (identity.get('publisher_sha256') and before != identity['publisher_sha256'])):
            raise ValueError('media differs from verified download receipt')
        horizon = max(q['query_time_s'] for q in groups[relative])
        actual = sampler.sample(path, horizon, 1, seconds=300)
        # Keep strict actual PTS admission used by the continuous scheduler.
        # The frozen sampler tolerates 1e-9 at its decode stopping boundary.
        frames = [frame for frame in actual['frames'] if frame['pts_s'] <= horizon]
        if sha(path) != before or actual['canvas_hw'] != [192, 256] or actual['fps'] != 1:
            raise ValueError('media identity or frozen sampler changed')
        snap, total = snapshots(frames, groups[relative])
        prefixes.update(snap)
        save(out / 'frames' / (entry['video_id'] + '.json'), frames)
        sources.append(dict(source_path=relative, video_id=entry['video_id'], source_bytes=path.stat().st_size,
                            source_sha256=before, max_query_time_s=horizon, sampled_frames=len(frames)))
        print(entry['video_id'], total, flush=True)
    save(out / 'sources.json', sources)
    save(out / 'snapshots.json', prefixes)
    save(out / 'manifest.json', {p.relative_to(out).as_posix(): sha(p)
                               for p in sorted(out.rglob('*.json'))})
