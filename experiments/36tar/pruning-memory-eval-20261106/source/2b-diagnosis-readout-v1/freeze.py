"""Pin executed project code, inputs and CPU parity; no resource submission."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];PREP=ROOT/'runs/diagnosis_readout_prepare_20261105_01'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    a,b=read(PREP/'CPU_local.json'),read(PREP/'CPU_cluster.json')
    assert a==b and a['passed'] and a['layouts']==88 and a['selected_pairs']==159
    files={}
    for name,h in read(PREP/'CPU_local.imports.json').items():
        if 'site-packages/' in name or name.startswith('.venv/'):continue
        canonical='repo/'+name[len('publish/qwen35-streaming-2b/'):] if name.startswith('publish/qwen35-streaming-2b/') else name
        assert sha(ROOT/name)==h,name
        files[canonical]=h
    for p in (ROOT/'experiments/2b-diagnosis-readout-v1').iterdir():
        if p.suffix in ('.py','.sh'):files[p.relative_to(ROOT).as_posix()]=sha(p)
    for name in ('questions.json','sources.json','plan.json','input_provenance.json','CPU_local.json','CPU_cluster.json'):
        p=PREP/name;files[p.relative_to(ROOT).as_posix()]=sha(p)
    for p in (ROOT/'models/Qwen3.5-2B').glob('*.json'):files[p.relative_to(ROOT).as_posix()]=sha(p)
    (PREP/'execution_lock.json').write_text(json.dumps(files,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Frozen',len(files),'execution files')
if __name__=='__main__':main()
