"""Build all fixed prompt layouts and exercise the real readout on a tiny CPU model."""
import hashlib
import json
import os
from pathlib import Path

from experiment import BanksForReadout, P, ROOT, OUT, VIEWS, ask, make_builder, question, read, runtime, sparse_pair, torch
from resolution_runtime import context

assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
if os.name != 'nt':
    os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:4]))
torch.set_num_threads(1)
torch.manual_seed(20261103)
env = runtime()
proc = env['AutoProcessor'].from_pretrained(ROOT / 'models/Qwen3.5-2B', local_files_only=True)
model = env['Qwen3_5ForConditionalGeneration'](
    env['Qwen3_5Config'](**env['tiny_config'](proc), attn_implementation='sdpa')).float().eval()
builder = make_builder(env, proc)
qs = read(P / 'questions.json')
layout_digest = hashlib.sha256()
layouts = pairs = tiny_qa = 0
with torch.inference_mode():
    for s in read(P / 'sources.json'):
        bank = BanksForReadout()
        trace = s['sampled_trace']
        for i, (a, b) in enumerate(zip(trace[::2], trace[1::2])):
            x = torch.randn(108, model.get_input_embeddings().embedding_dim)
            nodes = {}
            for j, view in enumerate(VIEWS):
                ids = torch.arange(108) if view == 'full' else (torch.arange(54) * 2 + j % 2)
                nodes[view] = sparse_pair(x[ids], ids, i, a['pts_s'], b['pts_s'])
            bank.append(nodes)
            pairs += 1
            for q in qs:
                if q['source_name'] != s['name'] or q['query_time_s'] != b['pts_s']:
                    continue
                text = question(q, 0, 'B')
                for reader in bank.readers.values():
                    limit = reader.banks['LS'].budget
                    _, pos, ids, meta = context(model, builder, reader.export('LS'), text, visual_limit=limit)
                    assert ids.shape[1] + 128 <= 4096 and meta['visual_tokens'] <= limit
                    assert all(n['end'] <= q['query_time_s'] for n in meta['nodes'])
                    layout_digest.update(ids.numpy().tobytes())
                    layout_digest.update(pos.numpy().tobytes())
                    layouts += 1
                    if tiny_qa < 10:
                        z = ask(env, model, builder, reader, 'LS', text, limit=1)
                        assert z['branch_released'] and z['read_state_unchanged']
                        tiny_qa += 1
assert (pairs, layouts, tiny_qa) == (429, 180, 10)
result = dict(passed=True, synthetic_pairs=pairs, layouts=layouts, tiny_QA=tiny_qa,
              layout_sha256=layout_digest.hexdigest(), GPU=0, pretrained_weights=0)
with (OUT / ('CPU_local.json' if os.name == 'nt' else 'CPU_cluster.json')).open('x', encoding='utf-8') as f:
    json.dump(result, f, indent=2)
print(json.dumps(result))
