"""Frozen old training recipe, with an exact content-selection initialization."""
import importlib.util,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/7b-content-residual-repair-v1'))
from residual import ContentResidual,encode
sys.path.insert(0,str(ROOT/'experiments/7b-trained-prune-completion-v1'))
import training
sys.path.insert(0,str(Path(__file__).parent))
from verify import verify

def main():
    verify()
    spec=importlib.util.spec_from_file_location('frozen_residual_training_recipe',ROOT/'experiments/7b-trained-prune-completion-v1/run_train.py')
    base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
    original=training.encode;calls=0
    def bounded_encode(*args,**kwargs):
        nonlocal calls
        calls+=1
        assert calls<=14517,'declared visual call budget exceeded'
        return encode(*args,**kwargs)
    try:
        training.encode=bounded_encode;base.GroupScorer=ContentResidual;base.verify=verify
        base.main()
        out=Path(sys.argv[sys.argv.index('--out')+1])
        identity=dict(architecture='ContentResidual',parameters_real_7b=87105,coverage='one per 2x2 tile plus global top fill',
                      base_recipe='7b-trained-prune-completion-v1',cut=16,retained=70,
                      checkpoint_selection='fixed final; no evaluation-selected checkpoint',
                      training_question_input_to_scorer=False,actual_visual_calls=calls)
        (out/'method.json').write_text(json.dumps(identity,indent=2)+'\n')
    finally:training.encode=original

if __name__=='__main__':main()
