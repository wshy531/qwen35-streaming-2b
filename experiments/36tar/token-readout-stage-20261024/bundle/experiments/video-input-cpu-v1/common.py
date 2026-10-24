"""CPU-only input validation constants and immutable artifact helpers."""
import hashlib,json,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/video_input_cpu_prepare_20261024_01'
HERE=Path(__file__).parent
sys.path.append(str(ROOT/'experiments/2b-natural-eval-v1'))
sys.path.append(str(ROOT/'experiments/2b-long-stream-v1'))
CONFIGS={'slice1':dict(thread_count=1,thread_type='SLICE'),'auto4':dict(thread_count=4,thread_type='AUTO')}


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False);stream.write('\n')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1048576),b''):h.update(chunk)
    return h.hexdigest()
