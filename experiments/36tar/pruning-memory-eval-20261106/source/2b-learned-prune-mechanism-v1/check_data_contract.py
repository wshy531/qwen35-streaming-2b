"""Offline validation of identities, labels, splits and acquisition integrity."""
import argparse
import hashlib
import importlib.util
import json
import tempfile
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/learned_prune_data_prepare_20261101_01'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=OUT / 'CPU_data_contract.json')
    args = parser.parse_args()
    read = lambda name: json.loads((OUT / name).read_text(encoding='utf-8'))
    sources, qs, plan = read('sources.json'), read('questions.json'), read('acquisition_plan.json')
    ids = {s['video_id']: s for s in sources}
    assert len(ids) == 240 and len(qs) == 1983
    assert plan['sources_sha256'] == hashlib.sha256((OUT / 'sources.json').read_bytes()).hexdigest()
    for s, p in zip(sources, plan['files']):
        assert all(p[k] == v for k, v in s.items())
        assert len(p['mirror_lfs_sha256']) == 64
    for q in qs:
        assert q['role'] == ids[q['video_id']]['role']
        assert q['options'][q['answer_index']] == q['answer_text']
        assert q['query_policy'] == 'full_clip_EOF' and q['sampled_evidence_verified'] is False
    # Negative check: payload equality requires all three identity checks.
    spec = importlib.util.spec_from_file_location('acquisition', Path(__file__).with_name('acquire_media.py'))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    payload = b'identity-control-not-media'
    row = dict(bytes=len(payload), mirror_lfs_sha256=hashlib.sha256(payload).hexdigest(),
               official_zip_crc32=f'{zlib.crc32(payload):08x}')
    with tempfile.TemporaryDirectory(prefix='prune-identity-', dir=OUT) as tmp:
        p = Path(tmp) / 'fixture.bin'; p.write_bytes(payload)
        assert module.verify(p, row)
        assert not module.verify(p, dict(row, bytes=len(payload)+1))
        assert not module.verify(p, dict(row, mirror_lfs_sha256='0'*64))
        assert not module.verify(p, dict(row, official_zip_crc32='00000000'))
        p.write_bytes(payload[:-1] + b'X')
        assert not module.verify(p, row)
    result = dict(status='passed', videos=240, questions=1983, pixel_audit=False,
                  checks=['source-role consistency', 'unique IDs', 'original label-option equality',
                          'pinned plan-source identity', 'size/SHA/CRC corruption rejection'],
                  GPU=0, media_downloads=0, training_updates=0)
    args.out.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
