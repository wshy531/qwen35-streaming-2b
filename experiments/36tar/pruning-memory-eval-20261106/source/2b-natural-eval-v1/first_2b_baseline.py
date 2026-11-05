"""Short real-video interface acceptance; not a memory benchmark or accuracy score."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import av
import numpy as np
from PIL import Image
import psutil
import torch
from transformers import AutoProcessor

REVISION = '15852e8c16360a2fea060d615a32b45270f8a8fc'
QUESTION = 'Describe the visible scene and activity briefly.'
SOURCES = ['cross_task/0mnuAqJCg5c.mp4', 'cross_task/3kIs5YyLd_M.mp4',
    'Ego4D/clips/33c3f556-cf99-45ae-9d01-9051b9b1f19e.mp4',
    'Ego4D/clips/17bb68e6-328e-428d-8f1c-9fbf8b2471f2.mp4']


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def tiny_config(processor):
    return dict(text_config=dict(vocab_size=248320,hidden_size=32,intermediate_size=64,
        num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=16,
        layer_types=['linear_attention','full_attention'],linear_key_head_dim=16,
        linear_value_head_dim=16,linear_num_key_heads=2,linear_num_value_heads=2,
        rope_parameters=dict(rope_type='default',rope_theta=10000.,partial_rotary_factor=1.,mrope_section=[2,3,3])),
        vision_config=dict(depth=1,hidden_size=32,intermediate_size=64,num_heads=2,
            patch_size=16,spatial_merge_size=2,temporal_patch_size=2,out_hidden_size=32,num_position_embeddings=16),
        image_token_id=processor.image_token_id,video_token_id=processor.video_token_id,
        vision_start_token_id=processor.tokenizer.convert_tokens_to_ids(processor.vision_start_token),
        vision_end_token_id=processor.tokenizer.convert_tokens_to_ids(processor.vision_end_token),
        tie_word_embeddings=False)


def tensor_digest(cache):
    result = {}
    for name in ('key_cache','value_cache','conv_states','recurrent_states'):
        values = getattr(cache,name)
        result[name] = []
        for value in values:
            if value is None:
                result[name].append(None)
            else:
                item = value.detach().contiguous().cpu()
                result[name].append(dict(shape=list(item.shape),dtype=str(item.dtype),
                    bytes=item.numel()*item.element_size(),sha256=hashlib.sha256(item.numpy().tobytes()).hexdigest()))
    return result


def frames_from(path, FramePacket, fit_canvas):
    frames = []
    deadline = time.monotonic()+30
    with av.open(str(path)) as c:
        stream = c.streams.video[0]
        stream.codec_context.thread_count = 1
        fps = float(stream.average_rate)
        origin = None
        due = 0.
        for index, frame in enumerate(c.decode(stream)):
            if time.monotonic()>deadline or index>1000:
                raise TimeoutError('Bounded four-frame decode exceeded')
            if frame.time is None:
                raise ValueError('Frame has no timestamp')
            if origin is None:origin=float(frame.time)
            pts = float(frame.time)-origin
            if pts+1e-9<due:continue
            image = frame.to_image()
            image.thumbnail((256,256),Image.Resampling.BICUBIC)
            rgb = fit_canvas(np.asarray(image),(192,256))
            frames.append(FramePacket(index,pts,rgb,index+1))
            due += 1.
            if len(frames)==4:break
        total_num_frames=stream.frames
    if len(frames)!=4:
        raise ValueError('Need four genuine frames per source')
    return frames,dict(fps=fps,total_num_frames=total_num_frames,
        frames_indices=[x.frame_id for x in frames])


def question_suffix(processor):
    template = processor.apply_chat_template([dict(role='user',content=[dict(type='video'),dict(type='text',text=QUESTION)])],
        tokenize=False,add_generation_prompt=True,enable_thinking=False)
    marker = processor.vision_start_token+processor.video_token+processor.vision_end_token
    return template,marker,processor.tokenizer(template.split(marker)[1],add_special_tokens=False,return_tensors='pt').input_ids


@torch.inference_mode()
def complete(model, processor, kwargs, max_new):
    # One full prefill, followed only by single-token continuation in the original implementation.
    result = model(**kwargs,use_cache=True,logits_to_keep=1)
    logits = result.logits
    first = logits[0,-1].detach().float().cpu().numpy().copy()
    cache = result.past_key_values
    length = kwargs['input_ids'].shape[1] if 'input_ids' in kwargs else kwargs['inputs_embeds'].shape[1]
    if 'position_ids' in kwargs:
        rope = int(kwargs['position_ids'][1:].max())+1
    else:
        rope = length+int(model.model.rope_deltas.reshape(-1)[0])
    eos = set([processor.tokenizer.eos_token_id])
    configured = model.generation_config.eos_token_id
    eos.update(configured if isinstance(configured,list) else [configured])
    generated = []
    for _ in range(max_new):
        token = int(logits[0,-1].argmax())
        generated.append(token)
        if token in eos:break
        pos = torch.tensor([[[length]],[[rope]],[[rope]],[[rope]]],device=logits.device)
        result = model(input_ids=torch.tensor([[token]],device=logits.device),position_ids=pos,
            cache_position=torch.tensor([length],device=logits.device),
            attention_mask=torch.ones((1,length+1),dtype=torch.long,device=logits.device),
            past_key_values=cache,use_cache=True,logits_to_keep=1)
        cache,logits = result.past_key_values,result.logits
        length += 1
        rope += 1
    return dict(answer=processor.tokenizer.decode(generated,skip_special_tokens=True),generated_ids=generated),first


@torch.inference_mode()
def matched_inputs(model,builder,frames,device):
    prefix = builder.prefix_ids.to(device)
    embeddings = [model.get_input_embeddings()(prefix)]
    positions = [torch.arange(prefix.shape[1],device=device)[None,None,:].expand(4,1,-1)]
    ids = [prefix]
    seen = rope = prefix.shape[1]
    grids = []
    for index in range(0,len(frames),2):
        inputs, record = builder.pair(frames[index],frames[index+1],seen,rope)
        inputs = inputs.to(device)
        features = torch.cat(model.model.get_video_features(inputs.pixel_values_videos,
            inputs.video_grid_thw,return_dict=True).pooler_output,0)
        embed = model.get_input_embeddings()(inputs.input_ids)
        embed = embed.masked_scatter((inputs.input_ids==builder.processor.video_token_id).unsqueeze(-1).expand_as(embed),features.to(embed))
        embeddings.append(embed)
        positions.append(inputs.position_ids)
        ids.append(inputs.input_ids)
        grids.append(record['grid'])
        seen += inputs.input_ids.shape[1]
        rope = int(inputs.position_ids[1:].max())+1
    suffix = builder.question_ids(QUESTION).to(device)
    embeddings.append(model.get_input_embeddings()(suffix))
    positions.append(torch.stack([torch.arange(seen,seen+suffix.shape[1],device=device),
        *[torch.arange(rope,rope+suffix.shape[1],device=device) for _ in range(3)]])[:,None,:])
    ids.append(suffix)
    return dict(inputs_embeds=torch.cat(embeddings,1),position_ids=torch.cat(positions,2)),dict(
        input_ids=torch.cat(ids,1)[0].cpu().tolist(),position_sha256=hashlib.sha256(torch.cat(positions,2).cpu().numpy().tobytes()).hexdigest(),
        pair_grids=grids,vision_tokens_per_pair=int(inputs.video_grid_thw[0,1:].prod())//builder.processor.video_processor.merge_size**2)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['preflight','gpu'],required=True)
    p.add_argument('--implementation',choices=['original','stream'],required=True)
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--media',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--preflight',type=Path)
    a = p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    active = a.repo/'experiments/36tar/streaming-learning/code/qwen35-streaming-s4'
    sys.path.insert(0,str(a.repo if a.implementation=='original' else active))
    sys.path.append(str(active))
    from qwen35_local import Qwen3_5Config,Qwen3_5ForConditionalGeneration
    from streaming.online import PairInputBuilder,OnlinePairSession,fp32_reference
    from streaming.window import FramePacket
    from streaming.preprocessing import fit_canvas
    torch.set_num_threads(1)
    os.environ['TOKENIZERS_PARALLELISM']='false'
    model_manifest = json.loads((a.model_dir/'DOWNLOAD_MANIFEST.json').read_text(encoding='utf-8-sig'))
    if model_manifest['revision']!=REVISION or model_manifest['model']!='Qwen/Qwen3.5-2B':
        raise ValueError('Pinned 2B assets required')
    source_file = Path(sys.modules[Qwen3_5ForConditionalGeneration.__module__].__file__).resolve()
    expected_parent = (a.repo/'qwen35_local' if a.implementation=='original' else active/'qwen35_local').resolve()
    if source_file.parent!=expected_parent:
        raise ValueError('Wrong model implementation imported')
    identity = dict(script_sha256=sha(__file__),model_manifest_sha256=sha(a.model_dir/'DOWNLOAD_MANIFEST.json'),
        model_source_sha256=sha(source_file),online_source_sha256=sha(active/'streaming/online.py'),
        hybrid_source_sha256=sha(active/'streaming/hybrid.py'),question=QUESTION,canvas=[192,256],
        source_sha256={s:sha(a.media/s) for s in SOURCES},max_new_tokens=16)
    device = 'cpu'
    if a.stage=='gpu':
        if not a.preflight or not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
            raise RuntimeError('GPU mode requires matching CPU proof and allocated Slurm GPU')
        proof = json.loads((a.preflight/'summary.json').read_text(encoding='utf-8'))
        if proof['stage']!='preflight' or proof['identity']!=identity or proof['implementation']!=a.implementation:
            raise ValueError('Preflight inputs or source changed')
        free,total = torch.cuda.mem_get_info()
        guard=dict(free_gpu_bytes=free,total_gpu_bytes=total,available_ram_bytes=psutil.virtual_memory().available,
                   slurm_job_id=os.environ['SLURM_JOB_ID'])
        save(a.out/'resource_guard.json',guard)
        if free<12*2**30 or guard['available_ram_bytes']<12*2**30:
            raise MemoryError('12 GiB GPU/RAM load guard failed')
        device='cuda'
        model=Qwen3_5ForConditionalGeneration.from_pretrained(a.model_dir,local_files_only=True,
            dtype=torch.float32,attn_implementation='sdpa').to(device).eval()
        torch.cuda.reset_peak_memory_stats()
    else:
        torch.manual_seed(42)
        processor=AutoProcessor.from_pretrained(a.model_dir,local_files_only=True)
        model=Qwen3_5ForConditionalGeneration(Qwen3_5Config(**tiny_config(processor),attn_implementation='sdpa')).eval()
    processor=AutoProcessor.from_pretrained(a.model_dir,local_files_only=True)
    builder=PairInputBuilder(processor,canvas=(192,256),resize_mode='letterbox')
    records=[]
    report=dict(stage=a.stage,implementation=a.implementation,identity=identity,records=records,
        scope='four genuine first frames, two queries per source; interface evidence only, no benchmark accuracy',
        pretrained_weights_loaded=a.stage=='gpu')
    started=time.monotonic()
    with fp32_reference(),torch.inference_mode():
        for source_number,source in enumerate(SOURCES):
            frames,metadata=frames_from(a.media/source,FramePacket,fit_canvas)
            frame_identity=[dict(id=f.frame_id,pts=f.timestamp,sha256=hashlib.sha256(f.rgb.tobytes()).hexdigest()) for f in frames]
            session = (OnlinePairSession(model,processor,max_tokens=512,max_question_tokens=64,max_answer_tokens=16,
                resize_mode='letterbox',video_logits='none') if a.implementation=='stream' else None)
            for count in (2,4):
                if time.monotonic()-started>300:
                    raise TimeoutError('Five-minute inference watchdog')
                subset=frames[:count]
                common=dict(source=source,prefix_frames=count,frames=frame_identity[:count])
                if session:
                    for f in frames[count-2:count]:session.append(f)
                    before=tensor_digest(session.main.cache)
                    answer=session.ask(QUESTION,16,return_evidence=True)
                    first=answer.pop('evidence')['initial_logits'][0,-1].float().cpu().numpy().copy()
                    unchanged=before==tensor_digest(session.main.cache)
                    if not unchanged:raise AssertionError('Question mutated main video state')
                    row=dict(**common,mode='stream',main_unchanged=unchanged,main_status=session.status(),
                        answer=answer['answer'],generated_ids=answer['generated_ids'],input_ids=
                        builder.prefix_ids[0].tolist()+sum([item['token_ids'] for item in session.ledger],[])
                        +builder.question_ids(QUESTION)[0].tolist())
                    tag=f'{source_number}-{count}-stream'
                    np.save(a.out/(tag+'.npy'),first)
                    row['logits_file']=tag+'.npy'
                    records.append(row)
                else:
                    kwargs,protocol=matched_inputs(model,builder,subset,device)
                    answer,first=complete(model,processor,kwargs,16)
                    tag=f'{source_number}-{count}-matched'
                    np.save(a.out/(tag+'.npy'),first)
                    records.append(dict(**common,mode='matched',protocol=protocol,**answer,logits_file=tag+'.npy'))
                    del kwargs
                    template,marker,_=question_suffix(processor)
                    meta=dict(metadata,frames_indices=metadata['frames_indices'][:count])
                    native=processor(text=[template],videos=[np.stack([f.rgb for f in subset])],
                        video_metadata=[meta],do_resize=False,do_sample_frames=False,return_tensors='pt').to(device)
                    # The processor returns one [T,H,W] video grid; the pinned
                    # Qwen3.5 position oracle consumes one grid per time marker.
                    # Expand only the oracle grid, leaving the vision grid intact.
                    oracle_grid=torch.repeat_interleave(native.video_grid_thw,native.video_grid_thw[:,0],0).clone()
                    oracle_grid[:,0]=1
                    coordinates,delta=model.model.get_rope_index(**dict(native,video_grid_thw=oracle_grid))
                    native['position_ids']=torch.cat((torch.arange(native.input_ids.shape[1],device=device)[None,None,:],coordinates),0)
                    model.model.rope_deltas=delta
                    answer,first=complete(model,processor,dict(native),16)
                    tag=f'{source_number}-{count}-native'
                    np.save(a.out/(tag+'.npy'),first)
                    records.append(dict(**common,mode='native',input_ids=native.input_ids[0].cpu().tolist(),
                        video_grid=native.video_grid_thw.cpu().tolist(),
                        position_adapter='official_get_rope_index_with_per_temporal_group_grid',**answer,logits_file=tag+'.npy'))
                save(a.out/'summary.json',report)
            if session:session.close()
    report.update(status='complete',wall_seconds=time.monotonic()-started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None,
        peak_reserved_bytes=torch.cuda.max_memory_reserved() if device=='cuda' else None)
    save(a.out/'summary.json',report)
    print(json.dumps(dict(status='complete',stage=a.stage,implementation=a.implementation,answers=len(records),
                          pretrained_weights_loaded=report['pretrained_weights_loaded'],wall_seconds=report['wall_seconds'])))


if __name__=='__main__':main()
