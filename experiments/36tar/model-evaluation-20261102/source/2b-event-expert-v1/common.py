"""Official original-backend paired inputs, small low-rank adapters and event task."""
import hashlib
import json
from pathlib import Path
import sys
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[2]
PREP=ROOT/'runs/event_expert_prepare_20261026_01'
DATA=ROOT/'datasets/EPIC-event-expert-v1'
TEACHER=ROOT/'models/VideoMAE-base-SSV2'
STUDENT=ROOT/'models/Qwen3.5-2B'
sys.path.insert(0,str(ROOT/'experiments/2b-native-audit-v2'))
from audit_common import paired_inputs,official_identity
sys.path.insert(0,str(ROOT/'experiments/2b-event-supervision-v1'))
from collation import supervised_pair_inputs
LABELS=('take','put','open','close','insert','remove','hold')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:
        json.dump(data,stream,indent=2,ensure_ascii=False,
                  default=lambda x:sorted(x) if isinstance(x,(set,frozenset)) else str(x))


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def prompt(row):
    noun=row['noun'].replace(':',' ')
    return ('Watch the completed video clip. The target object is '+noun+'. '
            'Which action is performed on the target object? '
            'Answer with exactly one action name from: '+', '.join(LABELS)+'.')


def canvas(frame):
    from PIL import Image
    image=frame.to_image()
    scale=min(384/image.width,288/image.height)
    image=image.resize((max(1,round(image.width*scale)),max(1,round(image.height*scale))),Image.Resampling.BICUBIC)
    result=Image.new('RGB',(384,288));result.paste(image,((384-image.width)//2,(288-image.height)//2))
    return result


def decode(row,check_pixels=True):
    import av
    import numpy as np
    specs=row['frames'];wanted={f['frame_id']:f for f in specs};images=[];origin=None
    with av.open(str(DATA/row['clip_path'])) as container:
        stream=container.streams.video[0];stream.codec_context.thread_count=1
        for index,frame in enumerate(container.decode(stream)):
            if origin is None:origin=float(frame.time)
            if index not in wanted:continue
            pts=float(frame.time)-origin;spec=wanted[index]
            assert abs(pts-spec['pts_s'])<1e-7 and pts<=row['duration_s']+1e-7
            image=canvas(frame)
            if check_pixels:assert hashlib.sha256(image.tobytes()).hexdigest()==spec['rgb_sha256']
            images.append(np.asarray(image).copy())
            if len(images)==len(specs):break
    assert len(images)==16
    return images


def supervised(processor,row,rgb):
    return supervised_pair_inputs(processor,rgb,[f['pts_s'] for f in row['frames']],prompt(row),row['label'])


def generation_inputs(batch,boundary):
    result={k:v for k,v in batch.items() if k not in ('labels','use_cache')}
    for k in ('input_ids','attention_mask','mm_token_type_ids'):result[k]=result[k][:,:boundary]
    assert result['input_ids'].shape[1]==boundary
    return result


class LowRankLinear(nn.Module):
    """Frozen base plus B(Ax), without materializing a dense delta weight."""
    def __init__(self,base,rank=8,alpha=16):
        super().__init__();self.base=base;self.scale=alpha/rank
        self.A=nn.Parameter(base.weight.new_empty(rank,base.in_features))
        self.B=nn.Parameter(base.weight.new_zeros(base.out_features,rank))
        nn.init.kaiming_uniform_(self.A,a=5**.5)
    def forward(self,x):return self.base(x)+F.linear(F.linear(x,self.A),self.B)*self.scale


def attach(net,rank=8,alpha=16,expected=12):
    net.requires_grad_(False);names=[]
    for name,layer in list(net.named_modules()):
        if name.startswith('model.language_model.') and name.endswith(('.self_attn.q_proj','.self_attn.v_proj')):
            assert isinstance(layer,nn.Linear)
            parent,key=name.rsplit('.',1)
            setattr(net.get_submodule(parent),key,LowRankLinear(layer,rank,alpha));names.append(name)
    assert len(names)==expected
    return names


def adapter_state(net):return {n:p.detach().cpu().clone() for n,p in net.named_parameters() if p.requires_grad}


def restore(net,state):
    parameters=dict(net.named_parameters())
    assert {n for n,p in parameters.items() if p.requires_grad}==set(state)
    with torch.no_grad():
        for n,value in state.items():parameters[n].copy_(value)


def training_forward(net,batch,boundary,head=None,teacher_feature=None,weight=.1):
    target_count=batch['input_ids'].shape[1]-boundary
    args={k:v for k,v in batch.items() if k not in ('labels','use_cache')}
    output=net(**args,use_cache=False,output_hidden_states=head is not None,logits_to_keep=target_count+1)
    # Predict the first answer token from boundary-1, through the answer terminator.
    task=F.cross_entropy(output.logits[:,:-1].reshape(-1,output.logits.shape[-1]),
                         batch['input_ids'][:,boundary:].reshape(-1))
    auxiliary=task.new_zeros(())
    if head is not None:
        assert teacher_feature is not None
        video_id=net.config.video_token_id
        positions=(batch['input_ids'][0,:boundary]==video_id).nonzero().flatten()
        grid=batch['video_grid_thw'][-1]
        n=int(grid.prod())//net.config.vision_config.spatial_merge_size**2
        selected=positions[-n:]
        assert selected.numel()==n and int(selected.max())<boundary
        representation=output.hidden_states[-1][:,selected,:].mean(1)
        auxiliary=(1-F.cosine_similarity(head(representation),teacher_feature,dim=-1)).mean()
    return task+weight*auxiliary,task,auxiliary


def parsed_action(text):
    # Content parsing reported separately from exact formatting; no hidden synonym tuning.
    import re
    words=re.findall(r'[a-z-]+',text.lower())
    matches=[word for word in words if word in LABELS]
    return matches[0] if len(matches)==1 else None
