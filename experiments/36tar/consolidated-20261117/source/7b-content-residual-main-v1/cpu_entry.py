"""Run exact entrypoint on tiny synthetic CPU cases and record imported files."""
import runpy,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).parent
mode=sys.argv.pop(1);assert mode in ('train','evaluate')
sys.path.insert(0,str(HERE));sys.argv[0]=str(HERE/f'run_{mode}.py')
runpy.run_path(sys.argv[0],run_name='__main__')
paths=set()
for m in list(sys.modules.values()):
    p=getattr(m,'__file__',None)
    if p:
        p=Path(p).resolve()
        if p.is_relative_to(ROOT) and 'experiments' in p.parts and p.suffix=='.py':paths.add(p.relative_to(ROOT).as_posix())
(ROOT/f'runs/7b_content_residual_main_prepare_20261113_01/imports-{mode}.json').write_text(json.dumps(sorted(paths),indent=2)+'\n')
