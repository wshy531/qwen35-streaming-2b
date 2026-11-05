"""Evaluate the two frozen utility heads on the existing selector QA panel."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/2b-evidence-selector-eval-v1'))
from selector_qa_common import PREP, read, save, sha, shared, media_path

HERE = Path(__file__).resolve().parent
HEADS = ROOT / 'runs/history-utility-fit-220511'
ARMS = ('recent', 'full', 'utility_visible', 'utility_blind')


def select(gate, model, builder, heads, nodes, task):
    # Use the same descriptor and question-only forward as training. Convert
    # the archived evaluation node type without changing its tensor values.
    from resolution_nodes import Node
    from utility_head import choose
    converted = [Node(**vars(n)) for n in nodes]
    features = gate.features(model, builder, converted[-4:], converted[:-4],
                            dict(id=task['id'], video_id=task['source_name'], question=task['question']))
    choices = gate.subsets(4)
    mask = gate.torch.zeros(1, len(choices), 4, dtype=gate.torch.bool)
    for i, subset in enumerate(choices):
        mask[0, i, list(subset)] = True
    inputs = [features[k][None] for k in ('query', 'recent', 'history', 'metadata', 'node_mask')]
    return {name: list(choices[choose(head(*inputs, mask, blind_history=name=='utility_blind')[0], choices)])
            for name, head in heads.items()}


def main(out, cpu_check):
    if out.exists() or not out.resolve().is_relative_to((ROOT/'runs').resolve()):
        raise ValueError('New output within runs required')
    if cpu_check:
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    else:
        assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_CPUS_PER_TASK') == '4'
    lock = read(HERE/'execution_lock.json')
    for path, expected in lock.items():
        assert sha(ROOT/path) == expected, path
    base, env, make_builder, Bank, distance, pair_node, digest, _ = shared()
    from canonical_prompt import question_prompt, rotate
    sys.path.insert(0, str(ROOT/'experiments/2b-history-utility-v1'))
    import score_gate as gate
    from utility_head import HistoryUtilityHead
    torch = env['torch']
    torch.set_num_threads(1)
    torch.manual_seed(20261104)
    processor = env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B', local_files_only=True)
    builder = make_builder(env, processor)
    plan = read(PREP/'plan.json')
    tasks = {t['id']: t for t in read(PREP/'requests.json')}
    checkpoints = {r['task_id']: r for r in plan['checkpoints']}
    refs = {(r['task_id'], r['arm'], r['shift']): r for r in read(PREP/'references_LOCAL_ONLY.json')}
    assert len(tasks)==len(checkpoints)==100 and tasks.keys()==checkpoints.keys() and len(refs)==640
    assert len(plan['sources'])==24 and sum(s['max_pairs'] for s in plan['sources'])==4414
    assert sum(t['role']=='current' for t in tasks.values())==12
    assert sum(t['role']=='old' for t in tasks.values())==6
    heads = {}
    for arm, file in [('utility_visible','content_visible.pt'), ('utility_blind','blind_history.pt')]:
        ck = torch.load(HEADS/file, map_location='cpu', weights_only=True)
        assert ck['epoch']==20 and ck['train_questions']==1114 and ck['blind_history']==(arm=='utility_blind')
        head = HistoryUtilityHead()
        head.load_state_dict(ck['state_dict'])
        heads[arm] = head.eval().requires_grad_(False)
    for source in plan['sources']:
        path = media_path(source)
        assert path.stat().st_size == source['bytes'] and sha(path) == source['sha256']
    train = read(ROOT/'runs/history_utility_prepare_20261104_01/train_manifest_LOCAL_ONLY.json')
    assert not ({s['sha256'] for s in plan['sources']} & {g['sha256'] for g in train})
    if cpu_check:
        import hashlib
        layouts = matches = 0
        for item in checkpoints.values():
            chunks = []
            for node in item['nodes']:
                label = (f"<{node['center']:.1f} seconds>" if node['first']==node['last'] else
                         f"<{node['begin']:.1f}-{node['end']:.1f} seconds, compressed memory>")
                chunks.append(builder.tokenize(label + processor.vision_start_token +
                              processor.video_token * node['tokens'] + processor.vision_end_token))
            task = tasks[item['task_id']]
            for shift in range(5):
                suffix = builder.question_ids(question_prompt(rotate(task, shift)[0]))
                for subset in gate.subsets(4):
                    ids = torch.cat([builder.prefix_ids, *[chunks[i] for i in subset], *chunks[-4:], suffix], 1)
                    assert ids.shape[1]+8 <= 8192
                    ref = refs.get((task['id'], 'recent' if not subset else 'full' if len(subset)==4 else '', shift))
                    if ref:
                        assert hashlib.sha256(ids.numpy().tobytes()).hexdigest() == ref['input_ids_sha256']
                        matches += 1
                    layouts += 1
        assert (layouts, matches)==(8000,640)
        config = env['Qwen3_5Config'](**env['tiny_config'](processor), attn_implementation='sdpa')
        model = env['Qwen3_5ForConditionalGeneration'](config).float().eval().requires_grad_(False)
        width = model.get_input_embeddings().embedding_dim
        nodes = [pair_node(torch.randn(108,width), i, 2.*i, 2.*i+1, (9,12)) for i in range(8)]
        # Exercise the real trained heads with correctly shaped synthetic inputs.
        mask = torch.zeros(1,16,4,dtype=torch.bool)
        for i, subset in enumerate(gate.subsets(4)):
            mask[0,i,list(subset)] = True
        for arm, head in heads.items():
            gain = head(torch.zeros(1,2048),torch.zeros(1,4096),torch.zeros(1,4,4096),
                        torch.zeros(1,4,8),torch.ones(1,4,dtype=torch.bool),mask,blind_history=arm=='utility_blind')
            assert torch.isfinite(gain).all()
        task = next(iter(tasks.values()))
        before = digest(nodes)
        from resolution_nodes import Node
        with env['fp32_reference']():
            gate.features(model,builder,[Node(**vars(n)) for n in nodes[-4:]],
                          [Node(**vars(n)) for n in nodes[:4]],dict(id=task['id'],video_id=task['source_name'],question=task['question']))
            for subset in ([],[0,1,2],[0,1,2,3]):
                base.infer(env,model,builder,nodes,question_prompt(rotate(task,0)[0]),subset,full=len(subset)>2)
        assert digest(nodes)==before
        save(out/'completed.json',dict(status='cpu_passed', layouts=layouts, endpoint_ids=matches, media=24, micro_QA=3))
        return
    assert torch.cuda.device_count()==1 and torch.cuda.mem_get_info()[0]>24*2**30
    assert env['psutil'].virtual_memory().available>12*2**30
    model, info = env['Qwen3_5ForConditionalGeneration'].from_pretrained(ROOT/'models/Qwen3.5-2B',
        local_files_only=True, dtype=torch.float32, device_map='cuda', attn_implementation='sdpa', output_loading_info=True)
    assert not any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
    model.eval().requires_grad_(False)
    out.mkdir()
    started = time.monotonic()
    count = dict(vision=0, answers=0, representations=0, reference_matches=0, tokens=0)
    def guard():
        if time.monotonic()-started>58*60 or env['psutil'].Process().memory_info().rss>20*2**30:
            raise RuntimeError('Fixed time/RSS limit')
    delivered = set()
    with torch.inference_mode(), env['fp32_reference'](), (out/'answers.jsonl').open('x',encoding='utf-8') as log:
        for source in plan['sources']:
            scheduled = [t for t in tasks.values() if t['source_name']==source['name']]
            bank = Bank('LS',108)
            previous = None
            for index,(a,b) in enumerate(base.sampled_pairs(media_path(source),source,guard)):
                value = base.project(env,model,builder,a,b)
                descriptor = value.mean(0)
                bank.append(pair_node(value,index,a.timestamp,b.timestamp,(9,12)),
                            starts_event=previous is None or distance(previous,descriptor)>.047536224126815796)
                previous = descriptor.clone()
                count['vision'] += 1
                for task in scheduled:
                    item = checkpoints[task['id']]
                    if item['committed_pairs']!=index+1:
                        continue
                    nodes = bank.nodes()
                    assert len(nodes)==8 and b.timestamp==item['cutoff_s'] and b.timestamp<=task['query_time_s']
                    for node, expected in zip(nodes,item['nodes'],strict=True):
                        meta = node.metadata()
                        assert all(meta[k]==expected[k] for k in ('first','last','begin','end','center','tokens','exact_pair'))
                    before = digest(nodes)
                    choices = dict(recent=[],full=[0,1,2,3])
                    choices.update(select(gate,model,builder,heads,nodes,task))
                    count['representations'] += 1
                    for shift in range(5):
                        prompt = question_prompt(rotate(task,shift)[0])
                        for arm in ARMS:
                            guard()
                            subset = choices[arm]
                            answer = base.infer(env,model,builder,nodes,prompt,subset,full=len(subset)>2)
                            ref = refs.get((task['id'],arm,shift))
                            if ref:
                                assert all(answer[k]==ref[k] for k in ('input_ids_sha256','position_ids_sha256','generated_ids'))
                                count['reference_matches'] += 1
                            log.write(json.dumps(dict(task_id=task['id'],arm=arm,shift=shift,**answer))+'\n')
                            count['answers'] += 1
                            count['tokens'] += len(answer['generated_ids'])
                    assert digest(bank.nodes())==before and task['id'] not in delivered
                    delivered.add(task['id'])
                    log.flush()
            print(json.dumps(dict(sources_done=len({tasks[q]['source_name'] for q in delivered}),**count)),flush=True)
    assert delivered==set(tasks) and count['tokens']<=16000
    assert (count['vision'],count['answers'],count['representations'],count['reference_matches'])==(4414,2000,100,640)
    save(out/'completed.json',dict(status='complete',**count,updates=0))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--cpu-check',action='store_true')
    args=parser.parse_args()
    main(args.out,args.cpu_check)
