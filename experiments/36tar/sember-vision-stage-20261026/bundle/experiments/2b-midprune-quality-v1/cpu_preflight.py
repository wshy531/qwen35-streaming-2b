"""Real processor + random tiny model; old storage/QA parity and every formal layout."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from quality_common import ROOT,OUT,ARMS,OLD_ARMS,KEEP,read,save,sha,runtime,lines,golden_path
from quality_banks import Banks,selected_node
from quality_session import Session
from resolution_runtime import make_builder,context
from resolution_source import CausalSource,PairDriver


def main(output):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
    env=runtime();torch=env['torch'];torch.set_num_threads(1);torch.manual_seed(20261026)
    from sember_banks import Banks as Old
    from sember_runtime import context as old_context
    old=Old();old.banks={a:b for a,b in old.banks.items() if a in OLD_ARMS.values()};banks=Banks()
    parity=0
    for i in range(96):
        x=torch.randn(108,32) if i%7==0 else x+.02*torch.randn(108,32)
        odd=torch.arange(1,108,2);old.append(x,i,4.*i,4.*i+2)
        banks.append(x,(x[odd],odd),i,4.*i,4.*i+2)
        for arm,oldarm in OLD_ARMS.items():
            a,b=banks.banks[arm].export(),old.banks[oldarm].export();assert len(a)==len(b)
            for left,right in zip(a,b):
                assert left.metadata()==right.metadata()
                for field in ('features','mass','coords'):assert torch.equal(getattr(left,field),getattr(right,field))
            parity+=1
        assert len({tuple(n.first for n in b.nodes()) for b in banks.banks.values()})==1
    for arm,b in banks.banks.items():assert b.status()['tokens']==8*KEEP[arm]
    before=banks.fingerprint();banks.banks[ARMS[-1]].export()[0].features.zero_();assert banks.fingerprint()==before
    solo=Banks((ARMS[-1],));solo.append(None,(x[odd],odd),0,0.,2.)
    assert solo.previous is None and solo.status()['shared_previous_bytes']==0
    change=Banks(('post_change54_8',));change.append(torch.zeros(108,2048),None,0,0.,2.)
    assert change.status()['shared_previous_bytes']==884736
    processor=env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True)
    cfg=env['tiny_config'](processor);cfg['vision_config']['depth']=4
    model=env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](**cfg,attn_implementation='sdpa')).float().eval()
    builder=make_builder(env,processor,(288,384));qa=[]
    with torch.inference_mode(),env['fp32_reference']():
        for arm,oldarm in OLD_ARMS.items():
            left=context(model,builder,banks.banks[arm].export(),'What is visible?',visual_limit=8*KEEP[arm])
            right=old_context(model,builder,old.banks[oldarm].export(),'What is visible?',visual_limit=8*KEEP[arm])
            assert all(torch.equal(a,b) for a,b in zip(left[:3],right[:3]))
        warm=read(OUT/'inputs/protocol.json')['warmup']
        session=Session(env,model,builder)
        driver=PairDriver(CausalSource(DATA_PATH(warm['video_id'])),session.append_pair)
        try:
            p=driver.advance(8.);assert p['pairs']==2 and p['pending_real_frames']==1
            assert session.visual_calls==4
            for arm in ARMS:
                before=session.bank.fingerprint();emitted=[];r=session.ask(arm,'What is visible?',emitted.append,max_new_tokens=2)
                assert session.bank.fingerprint()==before and r['read_state_unchanged'] and r['branch_released']
                assert r['generated_ids']==[e['token_id'] for e in emitted] and r['visual_tokens']==2*KEEP[arm]
                qa.append(dict(arm=arm,visual_tokens=r['visual_tokens'],input_tokens=r['input_tokens']))
            driver.advance(12.);assert session.pairs==3 and session.visual_calls==6
        finally:driver.close();session.close()
        from sember_protocol import prompt
        prior={(r['question_id'],r['arm']):r for r in lines(golden_path()) if r['arm'] in OLD_ARMS.values()}
        assert len(prior)==1434 and sha(golden_path())==read(OUT/'inputs/protocol.json')['golden_answers_SHA']
        layouts={};max_suffix=0
        for q in read(OUT/'inputs/requests.json'):
            qid=q['question_id'];base=prior[qid,OLD_ARMS[ARMS[0]]];layouts[qid]={}
            max_suffix=max(max_suffix,builder.question_ids(prompt(q)).shape[1])
            for arm in ARMS:
                indices=torch.arange(108) if KEEP[arm]==108 else odd
                # Original source times from frozen reference; values/selection scores are irrelevant to text IDs.
                nodes=[selected_node(torch.zeros(KEEP[arm],model.config.text_config.hidden_size),indices,
                    n['first'],n['begin'],n['end']) for n in base['nodes']]
                emb,pos,ids,meta=context(model,builder,nodes,prompt(q),visual_limit=8*KEEP[arm])
                assert pos.shape==(4,1,ids.shape[1]) and emb.shape[1]==ids.shape[1]
                ids_sha=hashlib.sha256(ids.numpy().tobytes()).hexdigest()
                if arm in OLD_ARMS:assert ids_sha==prior[qid,OLD_ARMS[arm]]['bounded_context_ids_sha256']
                layouts[qid][arm]=dict(bounded_context_ids_sha256=ids_sha,visual_tokens=meta['visual_tokens'],
                    input_tokens=meta['input_tokens'],source_pair_ids=[n.first for n in nodes])
        assert max_suffix<=512 and sum(map(len,layouts.values()))==1912
    layout_file=OUT/'inputs/expected_layouts.json'
    if layout_file.exists():assert read(layout_file)==layouts
    else:save(layout_file,layouts)
    result=dict(status='passed',old_storage_steps=parity,all_formal_layouts=1912,
        old_context_ID_layouts_equal=1434,synthetic_old_context_embeddings_positions_exact=True,
        all_arms_same_pair_coverage=True,exact_eight_pair_bound=True,QA_branches=qa,QA_resume=True,
        standalone_mid_no_full_previous=True,standalone_change_previous_bytes=884736,max_suffix=max_suffix,
        expected_layouts_SHA=sha(layout_file),pretrained_loads=0,GPU_jobs=0,
        no_capability_or_speed_claim=True)
    save(output,result);print({k:v for k,v in result.items() if k!='QA_branches'},flush=True)


def DATA_PATH(vid):
    from quality_common import DATA
    return DATA/'videos'/(vid+'.mp4')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
