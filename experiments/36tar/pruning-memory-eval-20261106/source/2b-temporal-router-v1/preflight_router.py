"""CPU: all prompt identities, balanced mapping, leakage and shared tiny prefill."""
import argparse
from pathlib import Path
import os
from collections import Counter
from copy import deepcopy
import math
from route_common import ROOT,HERE,PREP,INPUTS,read,save,sha,question_key
from temporal_policy import normalize,lexical_route,render,aggregate,route_question,STAGES
from route_runtime import runtime,tokenized,score_view


def mechanism_checks():
    for stage,states in STAGES.items():
        seen={s:Counter() for s,_ in states}
        for shift in range(len(states)):
            p=render('What is happening now?',stage,shift)
            for code,s in zip(p['codes'],p['semantic_order']):seen[s][code]+=1
        assert all(set(v.values())=={1} and len(v)==len(states) for v in seen.values())
    for support,current,expected in [('Historical',None,'full'),('Aggregation',None,'full'),
        ('Interaction',None,'full'),('Predictive',None,'recent'),('Current','Boundary','recent'),
        ('Current','Ongoing','recent'),('Current','Persistent','full'),('Current','Outcome','full')]:
        def fake(p):
            target=support if len(p['codes'])==5 else current
            return dict(candidate_logits=[3.0 if s==target else -1.0 for s in p['semantic_order']])
        result=route_question('What is the current result?',fake)
        assert result['route']==expected
        assert result['classifier_forwards']==(9 if support=='Current' else 5)
    assert route_question('What now?',lambda p:dict(candidate_logits=[0.0]*len(p['codes'])))['route']=='full'
    try:normalize(dict(question='What now?',correct_index=2))
    except TypeError:pass
    else:raise AssertionError('task metadata accepted as router input')
    assert lexical_route('How many items have I taken away so far?')['route']=='full'
    assert lexical_route('From where I am now, how do I get back to the first room?')['route']=='full'
    assert lexical_route('What is the person holding now?')['route']=='recent'
    # Averaged log-probabilities must map rotated code positions back to semantics.
    views=[]
    for shift in range(5):
        p=render('Earlier event?', 'support',shift)
        views.append(dict(semantic_order=p['semantic_order'],candidate_logits=[5.0 if s=='Historical' else 0.0 for s in p['semantic_order']]))
    assert aggregate('support',views)['state']=='Historical'
    broken=deepcopy(views);broken[0]['semantic_order'].reverse()
    for invalid in (views[:-1],broken,[dict(views[0],candidate_logits=[math.nan]*5),*views[1:]]):
        try:aggregate('support',invalid)
        except ValueError:pass
        else:raise AssertionError('bad balanced view accepted')
    from route_analysis import evaluate
    tasks=read(INPUTS/'evaluation_tasks_LOCAL_ONLY.json');pairs=read(INPUTS/'frozen_branch_outputs_LOCAL_ONLY.json')
    decisions={t['router_id']:'full' for t in tasks}
    for invalid in (pairs[:-1],pairs+[pairs[0]]):
        try:evaluate(decisions,tasks,invalid)
        except ValueError:pass
        else:raise AssertionError('partial outcome data accepted')
    from route_analysis import baseline_audit,development_decision
    baselines=baseline_audit()['policies']
    assert not development_decision(baselines['lexical'],baselines)['passed']
    assert development_decision(baselines['question_constant_oracle_DIAGNOSTIC'],baselines)['passed']
    return dict(all_eight_semantic_mappings=True,complete_balanced_cycles=True,
        exact_tie_falls_back_to_full=True,code_permutations_and_logprob_aggregation=True,
        invalid_logits_mapping_or_missing_views_rejected=True,question_only_metadata_rejected=True,
        simple_rule_handles_mixed_now_history=True,missing_duplicate_cached_outcomes_rejected=True,
        fixed_stop_and_advance_gate_negative_positive_controls=True)


def main(output,tiny):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='-1':raise RuntimeError('CPU-only preflight required')
    env=runtime();torch=env['torch'];torch.set_num_threads(1)
    processor=env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True)
    tokenizer=processor.tokenizer
    # Regression for the observed BatchEncoding/Tensor template return variants.
    class TensorTokenizer:
        def apply_chat_template(self,*args,**kwargs):return torch.tensor([[1,2,3]])
        def encode(self,code,**kwargs):return [ord(code)]
    compatibility,_=tokenized(TensorTokenizer(),render('What now?','support',0))
    assert compatibility.shape==(1,3)
    requests=read(INPUTS/'router_requests.json')
    assert len(requests)==113 and all(set(r)=={'id','question'} for r in requests)
    checks=mechanism_checks();layouts=[]
    checks['template_mapping_and_tensor_returns_supported']=True
    import router_worker
    from unittest.mock import patch
    forbidden=ROOT/'runs/CPU_must_not_start_GPU_temporal_router'
    assert not forbidden.exists()
    with patch.dict(os.environ,{'SLURM_JOB_ID':'','SLURM_CPUS_PER_TASK':'4'}):
        try:router_worker.main(forbidden,Path('not-a-real-lock.json'))
        except RuntimeError as e:assert 'Slurm' in str(e)
        else:raise AssertionError('GPU worker accepted non-Slurm invocation')
    assert not forbidden.exists()
    checks['non_Slurm_worker_rejected_before_pretrained_load']=True
    for request in requests:
        assert question_key(request['question'])==request['id']
        for stage,states in STAGES.items():
            for shift in range(len(states)):
                p=render(request['question'],stage,shift)
                ids,codes=tokenized(tokenizer,p)
                import hashlib
                layouts.append(dict(id=request['id'],stage=stage,shift=shift,
                    input_tokens=ids.shape[1],code_token_ids=codes,
                    input_ids_sha256=hashlib.sha256(ids.contiguous().numpy().tobytes()).hexdigest()))
    assert len(layouts)==1017
    actual=[]
    if tiny:
        torch.manual_seed(20261027)
        model=env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](
            **env['tiny_config'](processor),attn_implementation='sdpa')).float().eval()
        calls=[0];hook=model.model.visual.register_forward_pre_hook(lambda *args:calls.__setitem__(0,calls[0]+1))
        for stage,states in STAGES.items():
            for shift in range(len(states)):
                p=render(requests[0]['question'],stage,shift)
                a=score_view(env,model,tokenizer,p)
                actual.append(dict(stage=stage,shift=shift,**{k:v for k,v in a.items() if k!='seconds'}))
        repeated=score_view(env,model,tokenizer,render(requests[0]['question'],'support',0))
        assert repeated['candidate_logits']==actual[0]['candidate_logits']
        assert model.model.rope_deltas is None and calls[0]==0
        hook.remove()
        assert all(a['branch_released'] and a['generated_tokens']==0 for a in actual)
    result=dict(status='passed_CPU_temporal_router',question_texts=113,
        prompt_layouts=1017,maximum_prompt_tokens=max(r['input_tokens'] for r in layouts),
        checks=checks,layouts=layouts,tiny_shared_prefills=len(actual)+(1 if tiny else 0),
        tiny_results=actual,pretrained_loads=0,new_real_QA_answers=0,GPU_jobs=0,
        source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in HERE.iterdir() if p.suffix in ('.py','.md','.sh')},
        input_sha256={p.name:sha(p) for p in INPUTS.iterdir() if p.is_file()},
        runtime_file=env['actual_source'].relative_to(ROOT).as_posix())
    save(output,result)
    print({k:v for k,v in result.items() if k not in ('layouts','tiny_results','source_sha256','input_sha256')})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--tiny',action='store_true')
    a=p.parse_args();main(a.output,a.tiny)
