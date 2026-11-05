"""Full input layout coverage and six real tiny QA executions; CPU only."""
import argparse,os,json,hashlib
from protocol import *
from worker import records_for,answer

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
    env=runtime();t=env['torch'];t.set_num_threads(1);t.manual_seed(SEED)
    processor=env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True)
    cfg=env['tiny_config'](processor);cfg['vision_config']['depth']=24
    model=env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](**cfg,attn_implementation='sdpa')).float().eval().requires_grad_(False)
    builder=make_builder(env,processor);scorer=GroupScorer(model.config.vision_config.hidden_size)
    audit={r['video_id']:r for r in read(DATA/'CPU_media_audit.json')['rows']}
    train=read(PREP/'train.json');evaluation=read(PREP/'eval.json');digest=hashlib.sha256();layouts=0;maximum=0
    # Exact token and four-axis construction for every training/eval question.
    for q in train+evaluation:
        counts=[54] if q['role']=='train' else [108,54]
        shifts=[q['shift']] if q['role']=='train' else [0,2]
        for count in counts:
            records=[];ids=t.arange(108) if count==108 else t.arange(1,108,2)
            s=audit[q['video_id']]['samples']
            for i in range(0,len(s),2):
                records.append((i//2,s[i]['pts_s'],s[min(i+1,len(s)-1)]['pts_s'],
                                t.zeros((count,model.config.text_config.hidden_size)),ids))
            for shift in shifts:
                rq=rotate(q,shift);assert rq['options'][rq['answer_index']]==q['options'][q['answer_index']]
                _,positions,tokens=context(model,builder,records,question_text(rq))
                assert tokens.shape[1]+8<=4096 and builder.tokenize(chr(65+rq['answer_index'])).shape[1]==1
                digest.update(tokens.numpy().tobytes());digest.update(positions.numpy().tobytes())
                maximum=max(maximum,tokens.shape[1]);layouts+=1
    from transformers.feature_extraction_utils import BatchFeature
    batch=BatchFeature(dict(pixel_values_videos=t.randn(432,1536),video_grid_thw=t.tensor([[1,18,24]])))
    pairs=[(0.,.5,batch)];q=evaluation[0];calls=[]
    with t.inference_mode(),env['fp32_reference']():
        for arm in ARMS:
            records=records_for(model,pairs,arm,scorer)
            result=answer(env,model,builder,records,q,limit=1)
            assert len(records[0][3])==(108 if arm=='full' else 54)
            calls.append(dict(arm=arm,generated_tokens=len(result['generated_ids'])))
    # Rebuild outside inference mode and exercise the exact main-loop objective.
    with env['fp32_reference']():
        records=records_for(model,pairs,'trained',scorer)
        loss,_=task_loss(model,builder,records,question_text(q),chr(65+q['answer_index']));loss.backward()
    norm=float(t.sqrt(sum((p.grad**2).sum() for p in scorer.parameters() if p.grad is not None)))
    assert norm>0 and all(p.grad is None for p in model.parameters())
    result=dict(status='passed',unique_input_layouts=layouts,logical_eval_answers=7824,
        training_steps=1237,maximum_context_tokens=maximum,layout_sha256=digest.hexdigest(),
        tiny_QA=calls,scorer_gradient_norm=norm,previous_detached=True,GPU=0,pretrained_loads=0,
        random_weights=True,training_label_rotation_histogram={str(i):sum((q['answer_index']-q['shift'])%5==i for q in train) for i in range(5)})
    a.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))

if __name__=='__main__':main()
