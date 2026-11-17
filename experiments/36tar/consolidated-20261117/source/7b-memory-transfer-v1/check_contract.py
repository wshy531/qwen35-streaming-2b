"""Actual tokenizer/pixels + production readout checks; never load 7B weights."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
import transformers
from transformers import AutoProcessor
from native_cpu import make_tiny
from readout import Memory,ARMS,question,layout,context,generate_context,native_inputs,pair_inputs,answer
from q25_nodes import pair_node,digest
from run_cost import safe
ROOT=Path(__file__).resolve().parents[2]
PREP=ROOT/'runs/7b_transfer_prepare_20261107_01'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main(processor_dir,out):
    assert transformers.__version__=='5.2.0'
    torch.set_num_threads(1);torch.manual_seed(7007)
    processor=AutoProcessor.from_pretrained(processor_dir,local_files_only=True)
    model=make_tiny(real_tokens=True)
    decoder=read(PREP/'decoded_cpu.json');qa=read(PREP/'profile.json')[0]
    micro=[]
    with torch.no_grad():
        for row in decoder['rows']:
            assert sha(PREP/row['frames_file'])==row['frames_sha256']
            frames=np.load(PREP/row['frames_file'])['frames']
            native=native_inputs(processor,list(frames),question(qa,0,'B'))
            assert native['video_grid_thw'].tolist()==[[4,20,28]]
            memory=Memory();parts=[]
            for i in range(4):
                batch=pair_inputs(processor,frames[2*i:2*i+2])
                parts.append(batch['pixel_values_videos'])
                feature=model.get_video_features(**batch,return_dict=True).pooler_output[0]
                memory.append(feature,i,float(2*i),float(2*i+1))
            assert torch.equal(torch.cat(parts),native['pixel_values_videos'])
            nodes=memory.select('recent8frames',qa);emb,pos,ids=context(model,processor,nodes,question(qa,0,'B'))
            assert torch.equal(ids,native['input_ids'])
            expected,_=model.model.get_rope_index(ids,video_grid_thw=native['video_grid_thw'],second_per_grid_ts=native['second_per_grid_ts'])
            assert torch.equal(pos,expected)
            reference=model(**native,use_cache=False,logits_to_keep=1).logits[0,-1]
            generated,first=generate_context(model,emb,pos,limit=4)
            torch.testing.assert_close(reference,first,atol=1e-6,rtol=1e-5)
            model.model.rope_deltas=None
            native_generated=model.generate(**native,max_new_tokens=4,do_sample=False)[0,ids.shape[1]:].tolist()
            assert generated==native_generated
            for i in range(4,12):memory.append(torch.randn(140,64),i,float(2*i),float(2*i+1))
            before=digest(memory.ls.nodes())
            for arm in ARMS:
                response=answer(model,processor,memory,qa,arm,lambda:None,limit=2)
                assert len(response['generated_ids'])<=2 and response['read_state_unchanged']
            assert digest(memory.select('lexical',{'question':'What is visible right now?'}))==digest(memory.select('recent8frames',qa))
            assert digest(memory.select('lexical',{'question':'What happened before this?'}))==digest(memory.select('LS',qa))
            assert digest(memory.ls.nodes())==before and memory.ls.status()['tokens']<=1120
            micro.append(dict(key=row['key'],pixel_exact=True,position_exact=True,
                pixel_sha256=hashlib.sha256(native['pixel_values_videos'].numpy().tobytes()).hexdigest(),
                native_stream_max_abs=float((reference-first).abs().max()),ids=generated,arms=5))
    # The same code must accept the real 3584-wide 140-token features.
    wide=Memory()
    for i in range(10):wide.append(torch.randn(140,3584),i,float(2*i),float(2*i+1))
    assert wide.ls.status()['tokens']<=1120
    # All 228 formal layouts, including the real longest frame count; no labels.
    questions=read(PREP/'questions.json');traces=read(PREP/'traces.json');hashes=hashlib.sha256();maximum=0;count=0
    zero=torch.zeros(140,4);layouts=[]
    for q in questions:
        spec=traces[q['suite']+'|'+q['source_id']]
        if q['suite']=='D18':pairs=sum(r['pts_s']<=q['query_time_s']+1e-6 for r in spec['sampled_trace'])//2
        else:pairs=spec['states'][q['id']]['complete_pairs']
        assert pairs>=1
        all_nodes=[pair_node(zero,i,float(2*i),float(2*i+1),(10,14)) for i in range(pairs)]
        for arm in ARMS:
            # Exact endpoint counts; synthetic bounded layouts are not capability evidence.
            if arm=='complete_history':nodes=all_nodes
            elif arm=='recent8frames':nodes=all_nodes[-4:]
            elif arm=='recent16frames':nodes=all_nodes[-8:]
            else:nodes=all_nodes[:min(4,max(0,pairs-4))]+all_nodes[-4:] if pairs>4 else all_nodes
            ids,pos,_=layout(processor,nodes,question(q,0,'B'))
            hashes.update(ids.numpy().tobytes());hashes.update(pos.numpy().tobytes())
            maximum=max(maximum,ids.numel());count+=1
    assert count==1140
    json.dumps(safe({'expected_set':{'a','b'}}))
    result=dict(passed=True,version=transformers.__version__,real_weights=False,GPU=0,
        micro=micro,layout_count=count,layout_sha256=hashes.hexdigest(),max_input_tokens=maximum,
        full_width_bank_passed=True,fps=1.0,canvas_hw=[280,392],formal_questions=len(questions))
    out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--processor-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(a.processor_dir,a.out)
