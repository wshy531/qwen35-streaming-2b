"""Five same-coverage eight-pair banks. New encoder candidate, existing isolated QA."""
import sys,hashlib,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
for p in ['2b-midprune-quality-v1','2b-content-midprune-v1']:
    sys.path.insert(0,str(ROOT/'experiments'/p))
from quality_banks import Bank as ReferenceBank,selected_node
from resolution_session import Session as Reader
from content_prune import encode,CoverageScorer
import torch
MODES=('full','post_fixed','mid_fixed','post_content','mid_content')

class Banks:
    def __init__(self,modes):
        if not modes or len(set(modes))!=len(modes) or any(m not in MODES for m in modes):raise ValueError('fixed modes')
        self.banks={m:ReferenceBank('full108_8' if m=='full' else 'post_odd54_8') for m in modes}
        self.scorers={m:CoverageScorer() for m in modes if m.endswith('content')}
    def selector_digest(self):
        h=hashlib.sha256()
        for m,s in self.scorers.items():
            h.update(m.encode())
            if s.previous is not None:h.update(s.previous.cpu().numpy().tobytes())
        return h.hexdigest()
    def status(self):
        return {m:dict(**b.status(),selector_bytes=self.scorers[m].resident_bytes() if m in self.scorers else 0)
            for m,b in self.banks.items()}
    def clear(self):
        for s in self.scorers.values():s.reset()
        self.scorers.clear();self.banks.clear()

class Session(Reader):
    def __init__(self,env,model,builder,*,modes=MODES):
        if model.training or next(model.parameters()).dtype!=torch.float32 or builder.canvas!=(288,384):raise ValueError('fixed FP32 canvas')
        self.env,self.model,self.builder,self.mode=env,model,builder,'bounded'
        self.grid=(9,12);self.pairs=0;self.last_id=self.last_pts=None;self.invalid=self.tail=False
        self.bank=Banks(modes);self.visual_calls=0
        self.timing=dict(preprocess_seconds=0.,vision_seconds=0.,bank_or_main_update_seconds=0.)
    @torch.inference_mode()
    def append_pair(self,first,second,*,padding=False):
        if self.invalid or self.tail:raise RuntimeError('closed or finalized')
        if (self.last_id is not None and (first.frame_id<=self.last_id or first.timestamp<=self.last_pts)
            or (not padding and (first.frame_id>=second.frame_id or first.timestamp>=second.timestamp))
            or (padding and (first.frame_id!=second.frame_id or first.timestamp!=second.timestamp))):raise ValueError('ordered pairs')
        self.sync();start=time.monotonic()
        inputs,_=self.builder.pair(first,second,0,0,synthetic_padding=padding)
        inputs=inputs.to(self.model.get_input_embeddings().weight.device)
        self.sync();prepared=time.monotonic();saved=self.model.model.rope_deltas;encoded=prepared;bank_s=0.
        try:
            with self.env['fp32_reference']():
                modes=list(self.bank.banks);offset=self.pairs%len(modes);modes=modes[offset:]+modes[:offset]
                for mode in modes:
                    tick=time.monotonic()
                    if mode=='full':
                        value=torch.cat(self.model.model.get_video_features(inputs.pixel_values_videos,inputs.video_grid_thw,return_dict=True).pooler_output,0)
                        ids=torch.arange(108,device=value.device)
                    else:value,ids,_=encode(self.model.model.visual,inputs.pixel_values_videos,inputs.video_grid_thw,
                        mode=mode,scorer=self.bank.scorers.get(mode))
                    self.sync();encoded=time.monotonic();self.visual_calls+=1
                    node=selected_node(value,ids,self.pairs,first.timestamp,second.timestamp)
                    self.bank.banks[mode].append(node);self.sync();bank_s+=time.monotonic()-encoded
        except BaseException:self.invalid=True;raise
        finally:self.model.model.rope_deltas=saved
        self.timing['preprocess_seconds']+=prepared-start
        self.timing['vision_seconds']+=time.monotonic()-prepared-bank_s
        self.timing['bank_or_main_update_seconds']+=bank_s
        self.pairs+=1;self.last_id,self.last_pts=second.frame_id,second.timestamp;self.tail=padding
        return dict(pairs=self.pairs,visual_encodings=len(modes))
