import hashlib,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]
PREP=ROOT/'runs/7b_method_transfer_prepare_20261110_01'
sys.path.insert(0,str(ROOT/'experiments/7b-memory-transfer-v1'))
import readout
sys.path.insert(0,str(ROOT/'experiments/7b-time-protocol-v1'))
import segmented
from q25_nodes import Node,validate,digest
from q25_banks import Bank,distance
from video_source import CausalPairs
from vision import CoverageScorer,encode

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while b:=f.read(8*2**20):h.update(b)
    return h.hexdigest()
def tsha(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def path(q):
    p=Path(q['media_path']);return p if p.is_absolute() else ROOT/p

class Memory:
    def __init__(self,mode='full',archive=False):
        self.mode=mode;self.scorer=CoverageScorer() if mode=='content' else None
        n=140 if mode=='full' else 70
        # Equal pair capacity, not equal token budget: isolate representation loss.
        self.bank=Bank('LS',n,budget=8*n,recent=4*n,closed=3*n)
        self.recent16=Bank('M0',n,budget=8*n,recent=4*n,closed=3*n)
        self.previous=None;self.archive=[] if archive else None
    def append(self,features,ids,index,begin,end):
        ids=ids.cpu();x=features.detach().float().cpu()
        coords=torch.stack((torch.full_like(ids,index),ids//14,ids%14),-1)
        n=Node(x.clone(),torch.ones(len(ids),1),coords,index,index,float(begin),float(end),(begin+end)/2,grid=(10,14))
        validate(n);desc=x.mean(0)
        event=self.previous is None or distance(self.previous,desc)>readout.THRESHOLD
        self.bank.append(n,starts_event=event);self.recent16.append(n,starts_event=event)
        self.previous=desc.clone()
        if self.archive is not None:self.archive.append(n.copy())
    def parts(self):return [e.node for e in self.bank.recent],[e.node for e in self.bank.history]
    def select(self,route):
        if route=='recent8':return self.parts()[0]
        if route=='recent16':return self.recent16.nodes()
        if route=='LS':return self.bank.nodes()
        if route=='complete':assert self.archive is not None;return self.archive
        raise ValueError(route)

@torch.no_grad()
def answer(model,processor,nodes,q,limit=128):
    before=digest(nodes)
    emb,pos,ids=segmented.context(model,processor,nodes,readout.question(q,0,'B'),'segmented')
    generated,first=readout.generate_context(model,emb,pos,limit=limit)
    assert digest(nodes)==before
    return dict(text=processor.tokenizer.decode(generated,skip_special_tokens=True),generated_ids=generated,
        input_tokens=ids.numel(),visual_tokens=sum(len(n.features) for n in nodes),
        input_sha=tsha(ids),position_sha=tsha(pos),nodes=[n.metadata() for n in nodes])

def verify():
    for rel,h in read(PREP/'lock.json').items():assert sha(ROOT/rel)==h,rel

def load_model():
    import transformers
    from transformers import Qwen2_5_VLForConditionalGeneration,AutoProcessor
    assert transformers.__version__=='5.2.0' and torch.cuda.device_count()==1
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    m,info=Qwen2_5_VLForConditionalGeneration.from_pretrained(ROOT/'models/Qwen2.5-VL-7B-Instruct',local_files_only=True,
        dtype=torch.float32,device_map='cuda',attn_implementation='sdpa',output_loading_info=True)
    assert not any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
    assert sum(p.numel() for p in m.parameters())==8292166656
    return m.eval().requires_grad_(False),AutoProcessor.from_pretrained(ROOT/'models/Qwen2.5-VL-7B-Instruct',local_files_only=True)
