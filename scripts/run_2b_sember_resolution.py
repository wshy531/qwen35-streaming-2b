"""Resolution diagnostic: local plan, approved CPU/GPU, content-only score."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.wshy.online_memory_2b.isolation import configure
configure()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',required=True,choices=('local','plan','cpu','gpu','score'))
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--model-dir',type=Path)
    a=p.parse_args()
    if a.stage=='local':
        from experiments.wshy.sember_resolution_2b.local_checks import run
        run(a.out)
        return
    from experiments.wshy.sember_resolution_2b import runner
    if a.stage=='cpu':
        if a.model_dir is None: p.error('CPU stage requires --model-dir')
        runner.cpu(a.out,a.model_dir)
    else: getattr(runner,a.stage)(a.out)


if __name__=='__main__': main()
