"""Original isolated QA reader; only the incoming visual computation changes."""
import time
import torch
from quality_common import ARMS
from quality_banks import Banks
from resolution_session import Session as OriginalSession
from midprune import encode


class Session(OriginalSession):
    def __init__(self,env,model,builder,*,arms=ARMS):
        if model.training or next(model.parameters()).dtype!=torch.float32 or builder.canvas!=(288,384):
            raise ValueError('fixed FP32 eval/canvas required')
        self.env,self.model,self.builder,self.mode=env,model,builder,'bounded'
        self.grid=(9,12);self.pairs=0;self.last_id=self.last_pts=None;self.invalid=self.tail=False
        self.bank=Banks(arms);self.visual_calls=0
        self.timing=dict(preprocess_seconds=0.,vision_seconds=0.,bank_or_main_update_seconds=0.,
            full_vision_seconds=0.,mid_vision_seconds=0.)

    @torch.inference_mode()
    def append_pair(self,first,second,*,padding=False):
        if self.invalid or self.tail:raise RuntimeError('closed, failed or tail-finalized session')
        if (self.last_id is not None and (first.frame_id<=self.last_id or first.timestamp<=self.last_pts)
            or (not padding and (first.frame_id>=second.frame_id or first.timestamp>=second.timestamp))
            or (padding and (first.frame_id!=second.frame_id or first.timestamp!=second.timestamp))):
            raise ValueError('ordered real pair or explicit identical EOF padding required')
        self.sync();began=time.monotonic();inputs,record=self.builder.pair(first,second,0,0,synthetic_padding=padding)
        inputs=inputs.to(self.model.get_input_embeddings().weight.device);self.sync();prepared=time.monotonic()
        saved=self.model.model.rope_deltas;full=mid=None;full_s=mid_s=0.
        try:
            with self.env['fp32_reference']():
                modes=[]
                if any(a!='mid_odd54_8' for a in self.bank.banks):modes.append('full')
                if 'mid_odd54_8' in self.bank.banks:modes.append('mid')
                # Rotate the two shared quality encodings; standalone workers have one.
                if self.pairs%2:modes.reverse()
                for mode in modes:
                    self.sync();start=time.monotonic()
                    if mode=='full':
                        vision=self.model.model.get_video_features(inputs.pixel_values_videos,inputs.video_grid_thw,return_dict=True)
                        full=torch.cat(vision.pooler_output,0)
                        if full.shape!=(108,self.model.get_input_embeddings().embedding_dim):raise ValueError('unexpected full visual grid')
                    else:
                        x,ids,_=encode(self.model.model.visual,inputs.pixel_values_videos,inputs.video_grid_thw,mode='mid_odd')
                        mid=(x,ids)
                    self.sync();elapsed=time.monotonic()-start
                    if mode=='full':full_s+=elapsed
                    else:mid_s+=elapsed
                    self.visual_calls+=1
                encoded=time.monotonic();trace=self.bank.append(full,mid,self.pairs,first.timestamp,second.timestamp)
                self.sync();finished=time.monotonic()
        except BaseException:self.invalid=True;raise
        finally:self.model.model.rope_deltas=saved
        self.timing['preprocess_seconds']+=prepared-began;self.timing['vision_seconds']+=encoded-prepared
        self.timing['bank_or_main_update_seconds']+=finished-encoded
        self.timing['full_vision_seconds']+=full_s;self.timing['mid_vision_seconds']+=mid_s
        self.pairs+=1;self.last_id,self.last_pts=second.frame_id,second.timestamp;self.tail=padding
        return dict(pairs=self.pairs,record=record,trace=trace,visual_encodings=len(modes))
