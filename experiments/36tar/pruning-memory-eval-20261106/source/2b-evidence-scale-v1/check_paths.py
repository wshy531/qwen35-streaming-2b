"""Reproduce the real cluster alias failure before checking the canonical fix."""
import hashlib,json,os
from pathlib import Path
from common import R,P,save
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
alias=Path('/home/u2025010462/share/czy/swt/my_pape/qwen35-streaming-2b-36tar')
assert alias.resolve()==R.resolve()
from PIL import Image
checked=[]
for folder in ('evidence-scale-nextqa-220093','evidence-scale-stream-220094'):
    a=next((alias/'runs'/folder/'sampled_frames').glob('*.png'))
    old_error=False
    try:a.relative_to(R)
    except ValueError:old_error=True
    assert old_error
    canonical=a.resolve();rel=canonical.relative_to(R).as_posix()
    assert (R/rel).resolve()==canonical and (R/rel).read_bytes()==a.read_bytes()
    with Image.open(canonical) as im:
        assert im.size==(384,288);digest=hashlib.sha256(im.convert('RGB').tobytes()).hexdigest()
    checked.append(dict(original_failure_reproduced=True,relative_path=rel,rgb_sha256=digest))
save(P/'CPU_path_fix_REMOTE.json',dict(status='passed_real_saved_images',checks=checked,GPU=0,model_changes=0))
print(json.dumps(dict(checked=len(checked),status='passed')))
