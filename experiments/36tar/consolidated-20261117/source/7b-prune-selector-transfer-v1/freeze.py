import sys
from common import *
import preflight,run_prune,run_selector,fit_selector,prepare,analyze

def main():
    files=set(Path(m.__file__).resolve() for m in list(sys.modules.values()) if getattr(m,'__file__',None) and Path(m.__file__).resolve().is_relative_to(ROOT/'experiments'))
    files.update(Path(__file__).parent.glob('*.py'));files.update(Path(__file__).parent.glob('*.sh'))
    for name in ['evaluation.json','evaluation_keys_LOCAL_ONLY.json','fit_sources.json','traces.json','references.json','audit_ids.json','audit_references.json','plan.json']:
        files.add(PREP/name)
    files.add(ROOT/'runs/7b_transfer_prepare_20261107_01/config.json')
    save(PREP/'lock.json',{p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)})
    print(json.dumps(dict(files=len(files),lock_sha256=sha(PREP/'lock.json'))))
if __name__=='__main__':main()
