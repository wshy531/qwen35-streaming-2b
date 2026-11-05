"""Frozen repair evaluation on StreamingBench75 and current/history18.

Two separately budgeted invocations. No updates or label access in inference.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/2b-memory-prune-panel150-v1'))
from worker import (P, REF, Memory, V0Pairs, GroupScorer, CoverageScorer,
                    exact_pairs, runtime, make_builder, hard_encode, encode_pair,
                    question, read, sha, torch, ask, context, sparse_pair)
from resolution_nodes import digest
from response_v4 import parse
from complete_context import full_context
HERE=Path(__file__).resolve().parent
PREP=ROOT/'runs/prune_memory_completion_prepare_20261105_01'
CHECKPOINTS={'guarded':'runs/repair-guard-220499/final.pt',
             'initial':'runs/training-order-main-220483/initial.pt',
             'old_CE':'runs/hard-mask-main-220062/mask_ce_step1237.pt'}
STREAM_ARMS=('complete_history','guarded_LS','guarded_recent8','initial_LS','full_LS','full_recent8')
PAIR_ARMS=tuple(f'{v}_{p}' for v in ('full','fixed12','old_CE','initial','guarded') for p in ('recent8','LS'))


def save(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def lock_check():
    for rel,h in read(PREP/'execution_lock.json').items():
        p=ROOT/rel
        if rel.startswith('repo/') and not p.exists():p=ROOT/'publish/qwen35-streaming-2b'/rel[5:]
        assert sha(p)==h,rel


class Observe:
    def __init__(self,model,heads,panel):
        self.model,self.heads,self.panel=model,heads,panel
        self.views=('full','guarded','initial') if panel=='stream' else ('full','fixed12','old_CE','initial','guarded')
        self.memory={}
        for v in self.views:
            m=Memory(v if v in ('full','fixed12') else 'mask_ce')
            m.banks={k:b for k,b in m.banks.items() if k in ('LS','M0' if v=='full' else 'M0_same8')}
            self.memory[v]=m
        self.previous=dict.fromkeys(heads)
        self.selector=CoverageScorer()
        self.archive=[]
        self.index=0

    def append(self,batch,begin,end):
        batch=batch.to(self.model.get_input_embeddings().weight.device)
        for view in self.views:
            if view in self.heads:
                value,ids,_,self.previous[view]=hard_encode(self.model.model.visual,batch.pixel_values_videos,
                    batch.video_grid_thw,self.heads[view],previous=self.previous[view],mode='gather')
            else:value,ids,_=encode_pair(self.model,batch,view,None,None,self.selector)
            self.memory[view].append(value,ids,self.index,begin,end)
            if view=='full' and self.panel=='stream':
                self.archive.append(sparse_pair(value,ids,self.index,begin,end))
        self.index+=1

    def close(self):
        for m in self.memory.values():m.close()
        self.archive.clear()
        self.previous.clear()


@torch.no_grad()
def full_answer(env,model,builder,nodes,prompt,guard,limit=128):
    before=digest(nodes)
    saved=model.model.rope_deltas
    branch=None
    try:
        emb,pos,ids,meta=full_context(model,builder,nodes,prompt,limit)
        branch=env['HybridSequence'](model,max_tokens=65536,session_id='unpruned-full-history')
        for begin in range(0,emb.shape[1],512):
            guard()
            end=min(begin+512,emb.shape[1])
            logits=branch.append(inputs_embeds=emb[:,begin:end],position_ids=pos[:,:,begin:end],
                logits_mode='last' if end==emb.shape[1] else 'none')
        eos=model.generation_config.eos_token_id
        stops=set(eos if isinstance(eos,list) else [] if eos is None else [eos])
        stops.add(builder.processor.tokenizer.eos_token_id)
        generated,reason=env['greedy_decode'](branch,logits,limit,stops)
    finally:
        if branch is not None:branch.cache.reset()
        model.model.rope_deltas=saved
    assert digest(nodes)==before
    return dict(text=builder.processor.tokenizer.decode(generated,skip_special_tokens=True),generated_ids=generated,
        reason=reason,read_state_unchanged=True,branch_released=True,
        input_ids_sha256=hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
        position_ids_sha256=hashlib.sha256(pos.cpu().numpy().tobytes()).hexdigest(),**meta)


def ask_arm(env,model,builder,obs,q,arm,guard,limit=128):
    prompt=question(q,0,'B')
    if arm=='complete_history':return full_answer(env,model,builder,obs.archive,prompt,guard,limit)
    view,policy=arm.rsplit('_',1)
    if policy=='recent8':policy='M0' if view=='full' else 'M0_same8'
    return ask(env,model,builder,obs.memory[view],policy,prompt,limit=limit)


def main(args):
    lock_check()
    auth=read(PREP/f'authorization_{args.panel}.json')
    assert auth['approved'] and auth['execution_lock_sha256']==sha(PREP/'execution_lock.json')
    local=read(PREP/'CPU_local.json');remote=read(PREP/'CPU_cluster.json')
    assert local['passed'] and remote['passed'] and remote['media']==77
    assert local['layout_sha256']==remote['layout_sha256']
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_CPUS_PER_TASK')=='4'
    assert args.out.resolve().is_relative_to((ROOT/'runs').resolve()) and not args.out.exists()
    args.out.mkdir()
    started=time.monotonic()
    budget=177 if args.panel=='stream' else 28
    count=dict(answers=0,reference_matches=0,pairs=0,vision=0,tokens=0,sources=0)
    refs={(r['panel'],r['id'],r['arm']):r for r in read(PREP/'references.json')}
    config=read(PREP/'plan.json')[args.panel]
    def guard():
        if time.monotonic()-started>budget*60:raise TimeoutError('fixed deadline, no automatic retry')
        if sum(p.stat().st_size for p in args.out.iterdir() if p.is_file())>128*2**20:raise RuntimeError('output budget')
    try:
        torch.set_num_threads(1)
        env=runtime()
        assert torch.cuda.device_count()==1
        model,info=env['Qwen3_5ForConditionalGeneration'].from_pretrained(ROOT/'models/Qwen3.5-2B',
            local_files_only=True,dtype=torch.float32,device_map='cuda',attn_implementation='sdpa',output_loading_info=True)
        assert not any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
        assert model.config.text_config.max_position_embeddings>=65536
        model.eval().requires_grad_(False)
        versions={n:p._version for n,p in model.named_parameters()}
        builder=make_builder(env,env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True))
        heads={}
        for name,path in CHECKPOINTS.items():
            if args.panel=='stream' and name=='old_CE':continue
            h=GroupScorer(model.config.vision_config.hidden_size).to('cuda')
            h.load_state_dict(torch.load(ROOT/path,map_location='cpu',weights_only=True))
            heads[name]=h.eval().requires_grad_(False)
        tasks=read(PREP/f'{args.panel}_questions.json')
        sources=read(PREP/f'{args.panel}_sources.json')
        prefix=read(REF)['prefixes'] if args.panel=='stream' else None
        arms=STREAM_ARMS if args.panel=='stream' else PAIR_ARMS
        with torch.no_grad(),env['fp32_reference'](),(args.out/'answers.jsonl').open('x',encoding='utf-8') as output:
            for source in sources:
                guard()
                path=ROOT/source['media_path'] if args.panel=='stream' else Path(read(P/'protocol.json')['media_root'])/source['name']
                assert sha(path)==source['media_sha256']
                obs=Observe(model,heads,args.panel)
                qs=sorted([q for q in tasks if q.get('source_id',q.get('source_name'))==source['name']],key=lambda q:(q['query_time_s'],q['id']))
                emitted=set()
                def respond(q):
                    # Reproduce old full/CE/fixed arms before new conclusions.
                    order=sorted(arms,key=lambda a:(args.panel,q['id'],a) not in refs)
                    for arm in order:
                        guard()
                        r=ask_arm(env,model,builder,obs,q,arm,guard)
                        assert all(n['end']<=q['query_time_s']+1e-6 for n in r['nodes'])
                        ref=refs.get((args.panel,q['id'],arm))
                        if ref:
                            for k in ('input_ids_sha256','position_ids_sha256','generated_ids','nodes'):
                                assert r[k]==ref[k],(q['id'],arm,k)
                            count['reference_matches']+=1
                        r.update(parse(r['text'],len(q['options']),'B'))
                        r.update(panel=args.panel,id=q['id'],arm=arm,source_id=source['name'],
                            query_time_s=q['query_time_s'],pairs=obs.index)
                        output.write(json.dumps(r,ensure_ascii=False)+'\n')
                        output.flush()
                        count['answers']+=1
                        count['tokens']+=len(r['generated_ids'])
                    emitted.add(q['id'])
                def commit(first,second,padding=False):
                    guard()
                    batch,_=builder.pair(first,second,0,0,synthetic_padding=padding)
                    obs.append(batch,first.timestamp,second.timestamp)
                    count['pairs']+=1
                    count['vision']+=len(obs.views)
                if args.panel=='stream':
                    driver=V0Pairs(path,commit,config='auto4')
                    try:
                        for q in qs:
                            state=driver.advance(q['query_time_s'])
                            assert state==prefix[q['id']]['state'] and obs.index==state['complete_pairs']
                            respond(q)
                    finally:driver.close()
                else:
                    for first,second in exact_pairs(env,path,source['sampled_trace']):
                        commit(first,second)
                        for q in qs:
                            if q['query_time_s']==second.timestamp:respond(q)
                assert emitted=={q['id'] for q in qs}
                obs.close()
                del obs
                gc.collect()
                torch.cuda.empty_cache()
                count['sources']+=1
                print(json.dumps(dict(seconds=time.monotonic()-started,**count)),flush=True)
        for k in ('answers','reference_matches','pairs','vision','sources'):assert count[k]==config[k],(k,count[k],config[k])
        assert count['tokens']<=config['max_generated_tokens']
        assert all(p._version==versions[n] and p.grad is None for n,p in model.named_parameters())
        lock_check()
        save(args.out/'completed.json',dict(status='completed',**count,seconds=time.monotonic()-started,
            base_updates=0,scorer_updates=0,peak_allocated=torch.cuda.max_memory_allocated()))
    except BaseException as exc:
        save(args.out/'failed.json',dict(error=repr(exc),**count))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--panel',choices=('stream','paired'),required=True)
    p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
