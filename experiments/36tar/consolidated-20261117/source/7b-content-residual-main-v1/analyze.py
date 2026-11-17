"""Use the unchanged scoring protocol; identify the new endpoints explicitly."""
import importlib.util,json,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
out=Path(sys.argv[sys.argv.index('--run')+1])
done=json.loads((out/'completed.json').read_text())
assert done['content_reference_matches']==840 and not done['cpu_synthetic']
spec=importlib.util.spec_from_file_location('frozen_paired_prune_analysis',ROOT/'experiments/7b-trained-prune-evaluation-v1/analyze.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base);base.main()
path=out/'analysis.json';report=json.loads(path.read_text(encoding='utf8'))
report['method_names']={'initial':'untrained content+coverage start','guard':'trained content residual; fixed final',
                        'full':'unpruned vision','content':'old untrained content reference','fixed':'fixed spatial pruning'}
old=ROOT/'runs/7b-trained-prune-eval-223786/analysis.json'
report['old_recipe_reference']=[r for r in json.loads(old.read_text(encoding='utf8'))['tables'] if r['method'].startswith(('initial_','guard_'))]
report['old_recipe_reference_sha256']=hashlib.sha256(old.read_bytes()).hexdigest()
report['limits']+=['Old random-start recipe reference is reused, not newly trained',
                   'Coverage rule and content start changed together; improvements do not isolate either component',
                   'Training acceptance is on training-source hard loss, not an unseen-quality guarantee']
path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
