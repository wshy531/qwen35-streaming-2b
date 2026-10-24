"""Build the approved six-arm E2 in exclusive new directories."""
import ast,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
NEW=ROOT/'experiments/2b-readout-compression-v1'
OUT=ROOT/'runs/2b_readout_compression_prepare_20261024_01'
OLD=ROOT/'experiments/2b-compressed-banks-v1'
def write(p,x):
    with p.open('x',encoding='utf-8',newline='\n') as f:f.write(x if isinstance(x,str) else json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def read(p):return json.loads(p.read_text(encoding='utf-8'))

def main():
    assert NEW.is_dir() and {p.name for p in NEW.iterdir()}=={'banks.py'}
    OUT.mkdir(exist_ok=False);(OUT/'inputs').mkdir()
    common=(OLD/'common.py').read_text(encoding='utf-8').replace('2b_compressed_banks_prepare_20261023_01','2b_readout_compression_prepare_20261024_01')
    start=common.index('ARMS =');end=common.index('\nfor relative',start)
    common=common[:start]+"POLICIES=('M2','A06','LS')\nARMS=tuple(p+'_'+mode for p in POLICIES for mode in ('full48','read24'))"+common[end:]
    common+="\nsys.path.append(str(ROOT/'experiments/2b-compressed-banks-v1'))\nCONFIGS={'slice1':dict(thread_count=1,thread_type='SLICE'),'auto4':dict(thread_count=4,thread_type='AUTO')}\n"
    write(NEW/'common.py',common)
    write(NEW/'io_source.py',(ROOT/'experiments/video-input-cpu-v1/source.py').read_text(encoding='utf-8'))
    text=(OLD/'session.py').read_text(encoding='utf-8').replace('nine_banks','readout_banks').replace('nine fixed banks required','six registered readouts required')
    text=text.replace('result = emit_decode(branch, logits, self.builder.processor, 8, stops, on_token)',
        'first_token=[]\n            def emit(item):\n                if not first_token:first_token.append(time.monotonic()-started)\n                on_token(item)\n            result = emit_decode(branch, logits, self.builder.processor, 8, stops, emit)')
    text=text.replace("nodes=meta['nodes'], context_build_seconds=","first_token_seconds=first_token[0], storage_node_digest=digest[arm.split('_')[0]], nodes=meta['nodes'], context_build_seconds=")
    text=text.replace('compressed-banks-qa','readout-compression-qa')
    ast.parse(text);write(NEW/'session.py',text)
    source=ROOT/'runs/2b_compressed_banks_prepare_20261023_01'
    for name in ('groups.json','questions.json','snapshots34.json'):shutil.copyfile(source/'inputs'/name,OUT/'inputs'/name)
    shutil.copyfile(source/'labels_LOCAL_ONLY.json',OUT/'labels_LOCAL_ONLY.json')
    references=[]
    for line in (source/'results-218058/answers.jsonl').read_text(encoding='utf-8').splitlines():
        r=json.loads(line)
        if r['arm'] in ('M2_full48','A06_full48','LS_full48'):
            references.append({k:r[k] for k in ('id','arm','generated_ids','input_ids_sha256','node_digest')})
    assert len(references)==909
    write(OUT/'historical_references_LOCAL_ONLY.json',dict(records=references))
    arms=[p+'_'+m for p in ('M2','A06','LS') for m in ('full48','read24')]
    write(OUT/'inputs/protocol.json',dict(arms=arms,groups=112,questions=303,measured_answers=1818,
        measured_visual_encodings=3981,warmup_visual_encodings=8,warmup_answers=6,
        answers_max=1824,generated_tokens_max=14592,visual_encodings=3989,pretrained_loads=1,
        decoder='auto4',bank_visual_budget=384,readout_visual_budget={'full48':384,'read24':192},
        max_new_tokens=8,dtype='FP32',no_main_KV=True,no_full_history_archive=True,
        promotion=dict(primary269_no_net_loss=True,supported16_no_net_loss=True,min_QA_reduction=0.15,
            priority=['LS','A06','M2']),
        budget=dict(partition='g078t',comment='gx_llm',GPUs=1,CPUs=4,RAM_GiB=24,minutes=45)))
    shell=(OLD/'run_eval.sh').read_text(encoding='utf-8').replace('2b-compressed-banks','2b-readout-compression').replace('2b_compressed_banks_prepare_20261023_01','2b_readout_compression_prepare_20261024_01').replace('01:00:00','00:45:00')
    write(NEW/'run_eval.sh',shell)
    # Retain the established content parser; change registered contrasts only.
    text=(OLD/'analyze.py').read_text(encoding='utf-8')
    text=text.replace("tuple((p+'_change24',p+'_full48') for p in ('M0','M2','A06','LS'))+(('M0_change24','M0_odd24'),)","tuple((p+'_read24',p+'_full48') for p in ('M2','A06','LS'))")
    text=text.replace('expected_answers=2727','expected_answers=1818').replace('expected=2727','expected=1818')
    ast.parse(text);write(NEW/'analyze.py',text)
    print(dict(stage='E2',arms=6,questions=303,answers=1818,visual_encodings=3989))

if __name__=='__main__':main()
