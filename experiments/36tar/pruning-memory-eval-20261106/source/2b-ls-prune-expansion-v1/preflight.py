"""Check frozen workload, max-sized QA layouts and real media before submission."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from worker import (AUDIT, CAPACITY, OLD, OUT, P, PANEL, REF, ROOT, VIEWS, V0Pairs,
                    ask, context, env_decode, groups, inputs, ls_only,
                    make_builder, question, read, runtime, sha, torch)

TRAIN = ROOT/'runs/learned_prune_main_prepare_20261101_01/train.json'
BASE = ROOT/'runs/evidence_scale_prepare_20261102_01/panel.json'


def workload():
    audits, prefixes = inputs()
    frozen = {q['id']: q for q in read(BASE)}
    train = {q['video_id'] for q in read(TRAIN)}
    n = 0
    for (dataset, vid), tasks in groups():
        for q in tasks:
            old = frozen[q['id']]
            for k in ('question', 'options', 'query_time_s', 'media_path', 'media_sha256', 'source_id'):
                assert q[k] == old[k], (q['id'], k)
            assert q['media_path'] == tasks[0]['media_path']
            assert q['media_sha256'] == tasks[0]['media_sha256']
        if dataset == 'NExT-QA':
            assert vid not in train and all(q['query_policy']=='full_clip_EOF' for q in tasks)
            n += (len(audits[vid]['samples'])+1)//2
        else:
            counts = [prefixes[q['id']]['state']['complete_pairs'] for q in tasks]
            assert counts == sorted(counts) and counts[0] > 0
            n += counts[-1]
    assert len(frozen) == 300 and len(groups()) == 222 and n == 26898
    assert sha(OLD) == read(OUT/'capacity_receipt.json')['sha256']['answers.jsonl']
    return n


def cpu(destination):
    n = workload()
    env = runtime()
    proc = env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B', local_files_only=True)
    model = env['Qwen3_5ForConditionalGeneration'](
        env['Qwen3_5Config'](**env['tiny_config'](proc), attn_implementation='sdpa')).float().eval()
    builder = make_builder(env, proc)
    checks, digest, max_tokens, tiny = 0, hashlib.sha256(), 0, 0
    for tasks, readers in (
            (read(PANEL), {v: ls_only(v) for v in VIEWS}),
            (read(P/'questions.json'), {f'{v}_{r}_{h}': ls_only(v, r, h)
                for v in VIEWS[1:] for r,h in CAPACITY})):
        for reader in readers.values():
            for i in range(20):
                ids = torch.arange(108) if reader.keep == 108 else torch.arange(0,108,2)
                x = torch.randn(len(ids), model.get_input_embeddings().embedding_dim)
                reader.append(x, ids, i, 2*i, 2*i+1)
        for q in tasks:
            prompt = question(q, 0, 'B')
            for reader in readers.values():
                _, pos, ids, meta = context(model, builder, reader.export('LS'), prompt,
                    visual_limit=reader.banks['LS'].budget)
                assert ids.shape[1]+128 <= 4096 and meta['visual_tokens'] <= reader.banks['LS'].budget
                digest.update(ids.numpy().tobytes())
                digest.update(pos.numpy().tobytes())
                max_tokens = max(max_tokens, ids.shape[1])
                checks += 1
        for reader in readers.values():
            answer = ask(env, model, builder, reader, 'LS', question(tasks[0],0,'B'),limit=1)
            assert answer['branch_released'] and answer['read_state_unchanged']
            tiny += 1
    assert (checks, tiny) == (1416,16)
    write(destination, dict(passed=True, synthetic_layouts=checks, layout_sha256=digest.hexdigest(),
          maximum_tokens=max_tokens, tiny_QA=tiny, main_pairs=n, training_video_id_overlap=0))


def write(path, result):
    with path.open('x',encoding='utf-8') as f:
        json.dump(result,f,indent=2)
    print(json.dumps(result), flush=True)


def media(destination):
    audits, prefixes = inputs()
    checked = 0
    sample = set()
    env = runtime()
    builder = make_builder(env, env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True))
    for (dataset, vid), tasks in groups():
        path = (ROOT/tasks[0]['media_path']).resolve(strict=True)
        assert path.is_relative_to(ROOT.resolve()) and sha(path)==tasks[0]['media_sha256']
        checked += 1
        if dataset in sample:
            continue
        if dataset == 'NExT-QA':
            assert len(env_decode(env,path,audits[vid],builder)) == (len(audits[vid]['samples'])+1)//2
        else:
            driver = V0Pairs(path, lambda a,b,padding: None, config='auto4')
            try:
                for q in tasks:
                    assert driver.advance(q['query_time_s']) == prefixes[q['id']]['state']
            finally:
                driver.close()
        sample.add(dataset)
    spec = read(P/'protocol.json')
    for row in read(P/'sources.json'):
        path = Path(spec['media_root'])/row['name']
        assert path.resolve().parent == Path(spec['media_root']).resolve() and sha(path)==row['media_sha256']
        checked += 1
    assert checked == 227 and len(sample)==2
    write(destination, dict(passed=True, files=checked, replayed_media_protocols=sorted(sample)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--media',action='store_true')
    a = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    if os.name != 'nt':
        os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:4]))
    torch.set_num_threads(1)
    torch.manual_seed(20261103)
    (media if a.media else cpu)(a.out)
