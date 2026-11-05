"""Six metadata-selected questions, unchanged evidence prompt, zero visual input."""
import argparse,json,os,time
import gpu_profile as s
from response_v4 import parse
def text_context(model,builder,records,question):
    t=s.runtime_torch
    ids=t.cat((builder.prefix_ids,builder.question_ids(question)),1)
    assert not (ids==builder.processor.video_token_id).any()
    assert not (ids==builder.processor.tokenizer.convert_tokens_to_ids(builder.processor.vision_start_token)).any()
    pos=t.arange(ids.shape[1]).expand(4,-1)[:,None]
    device=model.get_input_embeddings().weight.device
    return model.get_input_embeddings()(ids.to(device)),pos.to(device),ids
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=__import__('pathlib').Path,required=True);a=p.parse_args()
    assert os.environ.get('SLURM_JOB_ID') and not a.out.exists()
    for n,h in s.read(s.P/'no_visual_lock.json').items():assert s.sha(s.R/n)==h,n
    a.out.mkdir();start=time.monotonic();env=s.runtime();t=env['torch'];t.set_num_threads(1);assert t.cuda.device_count()==1
    s.runtime_torch=t;s.context=text_context
    model,info=env['Qwen3_5ForConditionalGeneration'].from_pretrained(s.R/'models/Qwen3.5-2B',local_files_only=True,dtype=t.float32,
        device_map='cuda',attn_implementation='sdpa',output_loading_info=True)
    assert not any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
    model.eval().requires_grad_(False);builder=s.make_builder(env,env['AutoProcessor'].from_pretrained(s.R/'models/Qwen3.5-2B',local_files_only=True))
    cases=s.read(s.P/'real_pilot_LOCAL_ONLY.json');seen=set();rows=[]
    with t.no_grad(),env['fp32_reference']():
        for case in cases:
            if case['question_id'] in seen:continue
            seen.add(case['question_id']);q=dict(question=case['question'],options=case['options'])
            result=s.generate(env,model,builder,None,q,'B');result.update(parse(result['text'],len(q['options']),'B'))
            rows.append(dict(result,id=case['question_id'],protocol='B',arm='no_visual_control',visual_tokens=0))
    assert len(rows)==6 and sum(len(r['generated_ids']) for r in rows)<=768
    (a.out/'answers.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    s.save(a.out/'completed.json',dict(status='complete_no_visual_control',answers=6,generated_tokens=sum(len(r['generated_ids']) for r in rows),
        visual_calls=0,visual_tokens=0,model_loads=1,seconds=time.monotonic()-start,API=0,training=0))
if __name__=='__main__':main()
