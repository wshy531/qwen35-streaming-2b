"""36 context-budget checks with synthetic features; one short random readout."""
import hashlib
import json
from pathlib import Path
import sys
import time

from prepare import ROOT,OUT,read,save


def synthetic(env,row):
    torch=env['torch'];legacy=env['load']();fs=row['committed_frames'];raw=[]
    for i in range(row['total_prefix_pairs']):
        raw.append(legacy.forest.raw(legacy.features.raw_block(torch.randn(48,32),i,fs[2*i]['pts_s'],fs[2*i+1]['pts_s'])))
    compressed=[]
    for m in row['expected_bank_nodes']:
        if m['exact_pair']:compressed.append(raw[m['first']].copy());continue
        source=m['surviving_source_pair_ids']
        coords=torch.tensor([[source[i%len(source)],i//8,i%8] for i in range(48)],dtype=torch.long)
        node=legacy.forest.Node(torch.randn(48,32),torch.full((48,1),float(m['original_pairs'])),
            coords,m['first'],m['last'],m['begin'],m['end'],m['center'],m['groups'],m['depth'])
        legacy.forest.validate(node);compressed.append(node)
    return raw,compressed


def main():
    start=time.monotonic();sys.path.insert(0,str(ROOT/'experiments/2b-natural-eval-v1'))
    import runtime_bridge
    runtime_bridge.REPO=ROOT/'publish/qwen35-streaming-2b'
    runtime_bridge.ACTIVE=runtime_bridge.REPO/'experiments/36tar/streaming-learning/code/qwen35-streaming-s4'
    env=runtime_bridge.runtime('memory');torch=env['torch'];torch.set_num_threads(1);torch.manual_seed(123)
    from diagnostic import choices,answer,LIMITS,ARMS
    from readout import context
    from score_memory_eval import prompt
    from online_bank import nodes_digest
    from capability_pilot import answer as frozen_answer
    build=env['PairInputBuilder'](env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',
        local_files_only=True),canvas=(192,256),resize_mode='letterbox')
    model=env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](**env['tiny_config'](build.processor))).float().eval()
    assert not torch.cuda.is_initialized();audits=[];forward=False;peak=0
    for row in read(OUT/'requests6_LOCAL.json'):
        raw,bank=synthetic(env,row);arms=choices(raw,bank,row);before={a:nodes_digest(ns) for a,ns in arms.items()}
        assert all(torch.equal(a.features,b.features) for a,b in zip(arms['C_raw_old_recent'][-4:],arms['D_compressed_old_recent'][-4:]))
        with torch.inference_mode(),env['fp32_reference']():
            for arm in ARMS:
                emb,pos,_,meta=context(model,build,arms[arm],prompt(row),limits=LIMITS,max_new_tokens=8)
                assert emb.shape[1]+8<=32768 and int(pos.max())<model.config.text_config.max_position_embeddings
                audits.append(dict(id=row['id'],arm=arm,input_tokens=meta['input_tokens'],visual_tokens=meta['visual_tokens'],nodes=len(arms[arm])))
                peak=max(peak,env['psutil'].Process().memory_info().rss);assert peak<1536*2**20
                del emb,pos
            if not forward:
                emitted=[];actual=answer(env,model,build,bank,prompt(row),emitted.append)
                expected=frozen_answer(env,model,build,bank,prompt(row))
                assert actual['generated_ids']==expected['generated_ids']==[e['token_id'] for e in emitted]
                forward=True
        assert all(nodes_digest(arms[a])==h for a,h in before.items())
        del raw,bank,arms
    assert len(audits)==36 and forward
    save('CPU_diagnostic_contexts36.json',dict(status='passed',contexts=audits,
        maximum_input_tokens=max(a['input_tokens'] for a in audits),maximum_visual_tokens=max(a['visual_tokens'] for a in audits),
        short_reader_matches_frozen_greedy=True,all_context_nodes_unchanged=True,
        C_D_recent_features_bitwise_equal=True,pretrained_model_loads=0,GPU_jobs=0,random_tiny_instances=1,
        peak_observed_RSS_bytes=peak,seconds=time.monotonic()-start,
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
        limitations=['synthetic32-wide feature layouts, not captured2048-wide bank semantics',
          'long contexts constructed only; one short random forward',
          'manual node selection is an oracle diagnostic']))
    print(dict(status='passed',contexts=36,max_input=max(a['input_tokens'] for a in audits),peak_RSS=peak,seconds=time.monotonic()-start),flush=True)


if __name__=='__main__':main()
