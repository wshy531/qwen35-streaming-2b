"""One predeclared 2B diagnostic and storage comparison; never submits jobs itself."""
import argparse
from collections import defaultdict
import gc
import hashlib
import json
import os
from pathlib import Path
import time
import bridge

HERE, ROOT, BASE = bridge.HERE, bridge.ROOT, bridge.BASE
from policies import CappedForest, ReservoirRecent, block
from prepare import ARMS, DIAG_ARMS

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def save(p, x):
    with Path(p).open('x', encoding='utf-8') as f:
        json.dump(x, f, ensure_ascii=False, indent=2)
        f.write('\n')

def verify(run):
    if not run.resolve().is_relative_to((ROOT/'runs').resolve()):
        raise ValueError('owned run directory required')
    p = read(run/'protocol.json')
    for name, field in [('requests.json','request_sha256'),
                        ('diagnostic_requests.json','diagnostic_request_sha256'),
                        ('source_lock.json','source_lock_sha256'),
                        ('dependency_lock.json','dependency_lock_sha256')]:
        if sha(run/name) != p[field]:
            raise ValueError('protocol input drift: '+name)
    for folder, lock in [(HERE, 'source_lock.json'), (BASE, 'dependency_lock.json')]:
        for rel, h in read(run/lock).items():
            if sha(folder/rel) != h:
                raise ValueError('runtime source drift: '+rel)
    rs = read(run/'requests.json')
    if len(rs) != 171 or len({r['id'] for r in rs}) != 171:
        raise ValueError('fixed171 required')
    for r in rs:
        if set(r) != {'id','cohort','media_path','media_sha256','model_visible'}:
            raise ValueError('worker request contains unexpected fields')
        if set(r['model_visible']) != {'question','options','query_time_s','frames'}:
            raise ValueError('answer-blind model-visible fields required')
    base = ROOT/'runs/2b-natural-eval-inputs-20261020-01'
    # This module has the same basename: load the prior check by file identity.
    import importlib.util
    spec = importlib.util.spec_from_file_location('locked_previous_evaluator', BASE/'evaluate.py')
    previous = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    previous.check(base)
    identity = read(base/'cpu_memory/preflight.json')['identity']
    repo = ROOT/'repo'
    active = repo/'experiments/36tar/streaming-learning/code/qwen35-streaming-s4'
    for path, field in [(repo/'qwen35_local/modeling_qwen3_5.py','original_model_sha256'),
                        (active/'qwen35_local/modeling_qwen3_5.py','stream_model_sha256'),
                        (active/'streaming/online.py','online_sha256'),
                        (active/'streaming/hybrid.py','hybrid_sha256'),
                        (ROOT/'models/Qwen3.5-2B/DOWNLOAD_MANIFEST.json','manifest_sha256')]:
        if sha(path) != identity[field]:
            raise ValueError('prior verified model/runtime identity drift')
    if sha(run/'baseline_timelines.json') != p['baseline_timelines_sha256']:
        raise ValueError('baseline trace changed')
    if rs != read(base/'requests.json'):
        raise ValueError('original171 request objects changed')
    original = {r['id']:r for r in rs}
    diagnostic = read(run/'diagnostic_requests.json')
    if len(diagnostic) != 9 or len({r['id'] for r in diagnostic}) != 9:
        raise ValueError('all9 required')
    if any(r['request'] != original[r['id']] for r in diagnostic):
        raise ValueError('manual diagnostic request drift')
    return p, rs, diagnostic

def banks(env, nodes):
    import fixed_stage2 as fixed
    exports, status, events = fixed.policies(env, nodes)
    candidate, uniform = CappedForest(), ReservoirRecent()
    for n, event in zip(nodes, events):
        candidate.append(block(n), starts_event=event['starts_event'])
        uniform.append(n)
    choices = dict(full_pairwise_reference=nodes, **exports,
                   B02_cap4=candidate.export(), reservoir4=uniform.export(), text_only=[])
    status.update(B02_cap4=candidate.status(), reservoir4=uniform.status())
    return choices, status, events

def diagnostic_choices(nodes, choices, first):
    old = [n.copy() for n in nodes[first:first+4]]
    recent = [n.copy() for n in nodes[-4:]]
    if len(old) != 4 or old[-1].last >= recent[0].first:
        raise ValueError('four old pairs before recent4 required')
    m2_old = choices['M2'][:-4]
    target = [n.copy() for n in m2_old if n.last >= first and n.first <= first+3]
    other = [n.copy() for n in m2_old if n.last < first or n.first > first+3]
    if not target:
        raise ValueError('forest provenance must cover target diagnostic interval')
    return dict(manual_old4=old, m2_target_roots=target,
                m2_target_roots_recent4=[*target,*recent],
                m2_all_old=[n.copy() for n in m2_old],
                manual_old4_other_old=sorted([*old,*other],key=lambda n:n.first))

def make_builder(env):
    return env['PairInputBuilder'](env['AutoProcessor'].from_pretrained(
        ROOT/'models/Qwen3.5-2B', local_files_only=True),canvas=(192,256),resize_mode='letterbox')

def cpu(run):
    p, rs, diagnostic = verify(run)
    env = bridge.runtime()
    torch = env['torch']
    torch.set_num_threads(1)
    torch.manual_seed(123)
    if torch.cuda.is_available():
        raise ValueError('CPU preflight must hide GPU')
    builder = make_builder(env)
    model = env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](
        **env['tiny_config'](builder.processor),attn_implementation='sdpa')).eval()
    from capability_pilot import answer
    from fixed_stage2 import node_digest
    timelines = read(run/'baseline_timelines.json')
    traces = {t['media']:t for t in timelines}
    groups = defaultdict(list)
    for r in rs:
        groups[r['media_path']].append(r)
    dmap = {r['id']:r for r in diagnostic}
    contexts, tiny, maximum = 0, 0, 0
    generated = set()
    summary = []
    with torch.inference_mode(), env['fp32_reference']():
        for media, rows in groups.items():
            fs = rows[0]['model_visible']['frames']
            nodes = []
            legacy = env['load']()
            for j in range(0,len(fs),2):
                a,b = fs[j],fs[min(j+1,len(fs)-1)]
                nodes.append(legacy.forest.raw(legacy.features.raw_block(
                    torch.randn(48,32),j//2,a['pts_s'],b['pts_s'])))
            choices, status, events = banks(env,nodes)
            for arm, ns in choices.items():
                if arm != 'full_pairwise_reference' and sum(len(n.features) for n in ns) > 384:
                    raise AssertionError('candidate exceeds384')
            # Actual previous boundary flags are replayed separately for structural diagnostics.
            capped = CappedForest()
            for n, event in zip(nodes,traces[media]['events']):
                capped.append(block(n),starts_event=event['starts_event'])
            summary.append(dict(media=media, replayed_old_boundary_cap=capped.status()))
            for r in rows:
                q = env['prompt'](r['model_visible'])
                selections = dict(choices)
                if r['id'] in dmap:
                    selections.update(diagnostic_choices(nodes,choices,dmap[r['id']]['old_first_pair']))
                for arm, ns in selections.items():
                    before = node_digest(ns)
                    if ns:
                        _,_,_,meta = env['context'](model,builder,ns,q,
                            limits=env['Limits'](8192,6144,128),max_new_tokens=8)
                        maximum = max(maximum,meta['input_tokens'])
                    if arm not in generated:
                        answer(env,model,builder,ns,q)
                        generated.add(arm)
                        tiny += 1
                    if before != node_digest(ns):
                        raise AssertionError('query mutation')
                    contexts += 1
            del nodes, choices
            gc.collect()
    if contexts != 1242:
        raise AssertionError('all1242 contexts required')
    save(run/'cpu_preflight.json',dict(status='passed',contexts=contexts,
         tiny_answers=tiny,max_input_tokens=maximum,pretrained_loads=0,
         protocol_sha256=sha(run/'protocol.json'),causal_cap_boundary_replay=summary))
    print(json.dumps(dict(status='passed',contexts=contexts,tiny_answers=tiny)),flush=True)

def worker(run,stage,out):
    p, rs, ds = verify(run)
    approval = read(run/'authorization.json')
    if approval.get('approved') is not True or approval.get('protocol_sha256') != sha(run/'protocol.json'):
        raise ValueError('specific user GPU approval receipt required')
    proof = read(run/'cpu_preflight.json')
    if proof['status'] != 'passed' or proof['protocol_sha256'] != sha(run/'protocol.json'):
        raise ValueError('matching CPU preflight required')
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('allocated Slurm GPU required')
    env = bridge.runtime()
    torch = env['torch']
    torch.set_num_threads(1)
    torch.manual_seed(123)
    free,_ = torch.cuda.mem_get_info()
    if free < 16*2**30 or env['psutil'].virtual_memory().available < 12*2**30:
        raise MemoryError('preload16GiB GPU/12GiB RAM guard')
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    model = env['Qwen3_5ForConditionalGeneration'].from_pretrained(
        ROOT/'models/Qwen3.5-2B',local_files_only=True,dtype=torch.float32,
        device_map='cuda',attn_implementation='sdpa').eval()
    if model.config.text_config.hidden_size != 2048 or next(model.parameters()).dtype != torch.float32:
        raise ValueError('2B FP32 required')
    save(out/'loaded.json',dict(loads=1,load_seconds=time.monotonic()-started))
    builder = make_builder(env)
    from capability_pilot import answer
    from fixed_stage2 import decode,encode,node_digest
    dmap = {r['id']:r for r in ds}
    groups = defaultdict(list)
    for r in rs:
        if stage == 'main' or r['id'] in dmap:
            groups[r['media_path']].append(r)
    written, generated_tokens, timelines = 0, 0, []
    with (out/'predictions.jsonl').open('x',encoding='utf-8') as f, torch.inference_mode(), env['fp32_reference']():
        for media, rows in groups.items():
            packets,_ = decode(env,rows)
            specs = rows[0]['model_visible']['frames']
            nodes = encode(env,model,builder,packets,specs)
            choices,status,events = banks(env,nodes)
            timelines.append(dict(media=media,states=status,events=events,
                                  bank_digests={a:node_digest(ns) for a,ns in choices.items()}))
            for r in rows:
                selections = (choices if stage == 'main' else
                    diagnostic_choices(nodes,choices,dmap[r['id']]['old_first_pair']))
                hashes = {a:node_digest(ns) for a,ns in selections.items()}
                for arm,ns in selections.items():
                    if time.monotonic()-started > p['worker_seconds']-30:
                        raise TimeoutError('fixed worker wall bound')
                    tick = time.monotonic()
                    try:
                        a = answer(env,model,builder,ns,env['prompt'](r['model_visible']))
                        item = dict(id=r['id'],cohort=r['cohort'],arm=arm,status='ok',**a)
                    except (ValueError,FloatingPointError) as exc:
                        item = dict(id=r['id'],cohort=r['cohort'],arm=arm,status='readout_error',
                                    text='',error=str(exc),generated_ids=[])
                    item.update(seconds=time.monotonic()-tick,
                                input_frames_sha256=hashlib.sha256(json.dumps(specs,sort_keys=True).encode()).hexdigest(),
                                node_metadata=[n.metadata() for n in ns])
                    f.write(json.dumps(item,ensure_ascii=False)+'\n')
                    f.flush()
                    written += 1
                    generated_tokens += len(item['generated_ids'])
                    if any(node_digest(selections[a]) != h for a,h in hashes.items()):
                        raise AssertionError('QA changed external bank')
                    if torch.cuda.memory_allocated() > 24*2**30 or env['psutil'].Process().memory_info().rss > 20*2**30:
                        raise MemoryError('allocation/RSS guard')
                    print(json.dumps(dict(stage=stage,done=written,arm=arm)),flush=True)
            del nodes, packets, choices, selections
            gc.collect()
            torch.cuda.empty_cache()
    if written != (1197 if stage == 'main' else 45):
        raise AssertionError('fixed complete denominator required')
    save(out/'timelines.json',timelines)
    save(out/'completed.json',dict(status='complete',stage=stage,records=written,
         generated_tokens=generated_tokens,model_loads=1,seconds=time.monotonic()-started,
         peak_allocated_bytes=torch.cuda.max_memory_allocated(),query_bank_isolation=True,
         protocol_sha256=sha(run/'protocol.json')))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',choices=['cpu','worker'])
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--stage',choices=['diagnostic','main'])
    parser.add_argument('--out',type=Path)
    args = parser.parse_args()
    if args.mode == 'cpu':
        cpu(args.run)
    else:
        worker(args.run,args.stage,args.out)
