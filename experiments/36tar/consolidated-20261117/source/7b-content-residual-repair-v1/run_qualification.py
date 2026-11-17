"""Reuse established QA/recompute tolerances, plus exact content initialization."""
import importlib.util,json,sys
from pathlib import Path
from residual import ROOT,ContentResidual,encode,torch
sys.path.insert(0,str(ROOT/'experiments/7b-trained-prune-completion-v1'))
import training
sys.path.insert(0,str(ROOT/'experiments/7b-prune-selector-transfer-v1'))
import vision

def main():
    spec=importlib.util.spec_from_file_location('frozen_base_qualification',ROOT/'experiments/7b-trained-prune-completion-v1/run_qualification.py')
    base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
    old_encode=training.encode;old_capture=training.capture;records=[]
    def capture(model,scorer,pairs,short,mode='mask'):
        c=old_capture(model,scorer,pairs,short,mode)
        # Compare identical gather execution for content initialization. The
        # inherited qualification separately compares mask/gather QA logits;
        # mixing their numerical paths here would repeat the old false alarm.
        if not records and mode=='gather':
            reference=vision.CoverageScorer();device=next(scorer.parameters()).device;d=0.
            with torch.no_grad():
                for i,(_,_,b) in enumerate(pairs):
                    x,ids,_=vision.encode(model.model.visual,b['pixel_values_videos'].to(device),
                        b['video_grid_thw'].to(device),mode='content',scorer=reference)
                    assert torch.equal(ids,c['cache'][i][1])
                    torch.testing.assert_close(x,c['cache'][i][2],atol=2e-5,rtol=5e-5)
                    d=max(d,float((x-c['cache'][i][2]).abs().max()))
            records.append(dict(initial_content_ids_equal=True,pairs=len(pairs),maximum_feature_difference=d))
        return c
    try:
        training.encode=encode
        base.GroupScorer=ContentResidual;base.encode=encode;base.capture=capture
        base.main()
        out=Path(sys.argv[sys.argv.index('--out')+1]);path=out/'completed.json'
        result=json.loads(path.read_text());result['content_initialization']=records[0]
        result['declared_upper_counts']=dict(vision=45,language=5,local_optimizer_steps=2,exported_updates=0)
        path.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(records[0]),flush=True)
    finally:
        training.encode=old_encode
        assert training.capture is old_capture

if __name__=='__main__':main()
