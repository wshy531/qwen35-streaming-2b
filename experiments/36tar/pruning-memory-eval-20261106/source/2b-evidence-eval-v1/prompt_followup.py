"""One predeclared formatting repair, same 12 sources; never replace v1 results."""
import argparse,gc,json,os,time
from pathlib import Path
import gpu_profile as old
from response_v2 import parse
INSTRUCTION='''Reply in exactly three short lines using these labels:
Answer: one option letter and its answer text.
Evidence: at most TWO short visible observations, with the supplied time labels if clear (at most 35 words total).
Support: one short sentence connecting those observations to your answer (at most 15 words).
Do not list rejected options or describe extra background. If evidence is insufficient, say so.
Distinguish visible facts from an inference. Do not invent audio, intentions or future events.'''
def followup_question(q,shift,protocol):
    assert protocol=='B' and shift==0
    return q['question']+'\n'+'\n'.join(f'{chr(65+i)}. {x}' for i,x in enumerate(q['options']))+'\n'+INSTRUCTION
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_CPUS_PER_TASK')=='4'
    P=old.P;R=old.R
    for n,h in old.read(P/'followup_lock.json').items():assert old.sha(R/n)==h,n
    assert not a.out.exists();a.out.mkdir();started=time.monotonic();answers=calls=tokens=0
    # Replace the experiment's prompt-builder binding, never a model kernel or old file.
    old.question=followup_question
    try:
        env=old.runtime();t=env['torch'];t.set_num_threads(1);assert t.cuda.device_count()==1
        model,info=env['Qwen3_5ForConditionalGeneration'].from_pretrained(R/'models/Qwen3.5-2B',local_files_only=True,dtype=t.float32,
            device_map='cuda',attn_implementation='sdpa',output_loading_info=True)
        assert not any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
        model.eval().requires_grad_(False);builder=old.make_builder(env,env['AutoProcessor'].from_pretrained(R/'models/Qwen3.5-2B',local_files_only=True))
        audit={r['video_id']:r for r in old.read(old.DATA/'CPU_media_audit.json')['rows']}
        with t.no_grad(),env['fp32_reference'](),(a.out/'answers.jsonl').open('x',encoding='utf-8') as stream:
            for q in old.read(P/'profile.json'):
                if time.monotonic()-started>12*60:raise TimeoutError('followup deadline')
                path=R/q['media_path'];assert old.sha(path)==q['media_sha256'];pairs=old.decode(env,path,audit[q['video_id']],builder)
                for arm in ('full','fixed12'):
                    records=old.records_for(model,pairs,arm);calls+=len(pairs)
                    result=old.generate(env,model,builder,records,q,'B');result.update(parse(result['text'],len(q['options']),'B'))
                    result.update(id=q['id'],arm=arm,protocol='B2');answers+=1;tokens+=len(result['generated_ids'])
                    stream.write(json.dumps(result,ensure_ascii=False)+'\n');stream.flush();del records;gc.collect()
                del pairs;t.cuda.empty_cache();print(json.dumps(dict(answers=answers,seconds=time.monotonic()-started)),flush=True)
        assert answers==24 and calls==360 and tokens<=3072
        old.save(a.out/'completed.json',dict(status='completed_single_prompt_repair',answers=answers,vision_calls=calls,generated_tokens=tokens,
                 seconds=time.monotonic()-started,API=0,model_loads=1,training=0,original_profile_preserved=True))
    except Exception as e:old.save(a.out/'failed.json',dict(error=repr(e),answers=answers));raise
if __name__=='__main__':main()
