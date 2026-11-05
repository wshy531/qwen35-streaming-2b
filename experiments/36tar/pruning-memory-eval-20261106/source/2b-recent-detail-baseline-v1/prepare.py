"""Freeze inherited questions and validate causal read layouts on CPU."""
import hashlib
import json
import os
import sys
from guard import *
sys.path.insert(0,str(ROOT/'experiments/2b-prune-checkpoints-v1'))
from worker import runtime, make_builder
sys.path.insert(0,str(ROOT/'experiments/2b-evidence-eval-v1'))
from response import question
from resolution_runtime import context


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    torch.set_num_threads(1)
    torch.manual_seed(20261103)
    source = ROOT/'runs/midprune_memory_prepare_20261102_01'
    target = ROOT/'runs/recent_detail_prepare_20261103_01'
    target.mkdir(parents=True,exist_ok=True)
    inputs = {}
    for name in ('questions.json','sources.json'):
        raw = (source/name).read_bytes()
        (target/name).write_bytes(raw)
        inputs[name] = hashlib.sha256(raw).hexdigest()
    qs = json.loads((target/'questions.json').read_text(encoding='utf-8'))
    sources = json.loads((target/'sources.json').read_text(encoding='utf-8'))
    assert len(qs)==18 and len(sources)==5 and len({q['id'] for q in qs})==18
    assert sum(q['role']=='current' for q in qs)==12 and sum(q['role']=='old' for q in qs)==6
    env = runtime()
    proc = env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True)
    model = env['Qwen3_5ForConditionalGeneration'](
        env['Qwen3_5Config'](**env['tiny_config'](proc),attn_implementation='sdpa')).float().eval().requires_grad_(False)
    builder = make_builder(env,proc)
    rows = []
    count = 0
    with torch.no_grad():
        for s in sources:
            memories = {arm:RecentDetail(arm) for arm in ARMS}
            trace = s['sampled_trace']
            for i in range(0,len(trace),2):
                first,second = trace[i:i+2]
                x = torch.randn(108,model.config.text_config.hidden_size)
                for memory in memories.values():
                    memory.append(x,i//2,first['pts_s'],second['pts_s'])
                count += 1
                for q in qs:
                    if q['source_name']!=s['name'] or q['query_time_s']!=second['pts_s']:
                        continue
                    assert digest(memories['uniform810'].export()[:-1])==digest(memories['protect864'].export()[:-1])
                    for arm,m in memories.items():
                        _,pos,ids,meta = context(model,builder,m.export(),question(q,0,'B'),
                            visual_limit=m.status()['total_visual_budget'],max_new_tokens=16)
                        assert ids.shape[1]+128<=4096
                        assert all(n['end']<=q['query_time_s'] for n in meta['nodes'])
                        rows.append(dict(id=q['id'],arm=arm,role=q['role'],input_tokens=ids.shape[1],
                                         visual_tokens=meta['visual_tokens'],future_evidence=False))
            for m in memories.values():
                m.close()
    assert count==429 and len(rows)==54 and len({(r['id'],r['arm']) for r in rows})==54
    result = dict(passed=True,inherited_input_sha256=inputs,source_count=5,questions=18,
                  current_questions=12,old_questions=6,layout_count=54,real_pair_timestamps=429,
                  synthetic_features=True,real_media_read=False,pretrained_weight_loads=0,GPU=0,
                  maximum_input_tokens=max(r['input_tokens'] for r in rows),output_reserve=128,
                  rows=rows)
    (target/'CPU_layout_local.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))


if __name__=='__main__':
    main()
