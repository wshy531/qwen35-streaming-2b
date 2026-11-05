"""Compare protocol identity and numerical continuation; no semantic accuracy claim."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original',type=Path,required=True)
    p.add_argument('--stream',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    original=json.loads((a.original/'summary.json').read_text(encoding='utf-8'))
    streaming=json.loads((a.stream/'summary.json').read_text(encoding='utf-8'))
    if original.get('status')!='complete' or streaming.get('status')!='complete' or original['stage']!=streaming['stage']:
        raise ValueError('Need complete runs of the same stage')
    for key in ('script_sha256','model_manifest_sha256','source_sha256','question','canvas','max_new_tokens'):
        if original['identity'][key]!=streaming['identity'][key]:
            raise ValueError('Different experiment identity: '+key)
    results=[]
    for stream in streaming['records']:
        matched=next(r for r in original['records'] if r['source']==stream['source']
            and r['prefix_frames']==stream['prefix_frames'] and r['mode']=='matched')
        native=next(r for r in original['records'] if r['source']==stream['source']
            and r['prefix_frames']==stream['prefix_frames'] and r['mode']=='native')
        if matched['frames']!=stream['frames'] or matched['protocol']['input_ids']!=stream['input_ids']:
            raise ValueError('Different actual frames or input tokens')
        x=np.load(a.original/matched['logits_file'],allow_pickle=False)
        y=np.load(a.stream/stream['logits_file'],allow_pickle=False)
        if x.shape!=y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('Invalid logit vectors')
        results.append(dict(source=stream['source'],prefix_frames=stream['prefix_frames'],
            actual_frames_and_protocol_ids_equal=True,max_abs_logit_error=float(np.abs(x-y).max()),
            logits_allclose_1e4=bool(np.allclose(x,y,atol=1e-4,rtol=1e-4)),
            main_unchanged=stream['main_unchanged'],generated_ids_equal=matched['generated_ids']==stream['generated_ids'],
            original_native_answer=native['answer'],original_matched_answer=matched['answer'],stream_answer=stream['answer'],
            native_and_matched_ids_equal=native['input_ids']==matched['protocol']['input_ids'],
            vision_tokens_per_pair=matched['protocol']['vision_tokens_per_pair']))
    if len(results)!=8:
        raise ValueError('Expected four sources with two prefixes each')
    report=dict(stage=original['stage'],pretrained_weights_loaded=original['pretrained_weights_loaded'],
        cases=results,passed=all(r['logits_allclose_1e4'] and r['main_unchanged'] and r['generated_ids_equal'] for r in results),
        scope='numerical interface acceptance only; no answer accuracy or memory advantage inferred')
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(stage=report['stage'],cases=len(results),passed=report['passed'],
        max_abs_logit_error=max(r['max_abs_logit_error'] for r in results),pretrained_weights_loaded=report['pretrained_weights_loaded'])))
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
