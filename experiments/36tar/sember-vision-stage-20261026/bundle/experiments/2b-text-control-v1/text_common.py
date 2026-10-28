"""Fixed missing-modality control; no weights loaded or resources allocated on import."""
from pathlib import Path
import hashlib
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/2b-sember-v1'))
from sember_common import read,save,sha,runtime
from sember_protocol import prompt,parse_choice,format_ok
from sember_runtime import make_builder
from sember_session import emit_decode,cache_bytes
OUT=ROOT/'runs/sember_text_control_prepare_20261025_01'
MAIN=ROOT/'runs/sember_main_prepare_20261025_01'


def text_ids(builder,q):
    question=prompt(q);rendered=builder.template(question)
    if rendered.count(builder.marker)!=1:raise ValueError('exactly one original video marker required')
    # Only remove the modality marker; keep the original MCQ and template text.
    rendered=rendered.replace(builder.marker,'',1)
    ids=builder.tokenize(rendered)
    forbidden={builder.processor.video_token_id,builder.processor.tokenizer.convert_tokens_to_ids(builder.processor.vision_start_token),
        builder.processor.tokenizer.convert_tokens_to_ids(builder.processor.vision_end_token)}
    assert not set(ids[0].tolist())&forbidden
    assert ids.shape[1]+16<=1024
    return ids,dict(input_ids=ids[0].tolist(),input_tokens=ids.shape[1],
        input_sha256=hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
        MCQ_text_sha256=hashlib.sha256(question.encode()).hexdigest(),
        question_suffix_sha256=hashlib.sha256(builder.question_ids(question).numpy().tobytes()).hexdigest())


def verify():
    lock=read(OUT/'inputs/execution_lock.json')
    for rel,h in lock['runtime_sha256'].items():
        path=ROOT/rel
        if rel.startswith('repo/') and not (ROOT/'repo').exists():path=ROOT/'publish/qwen35-streaming-2b'/rel[5:]
        assert sha(path)==h,rel
    for name,h in lock['input_sha256'].items():assert sha(OUT/'inputs'/name)==h,name
    return read(OUT/'inputs/protocol.json')
