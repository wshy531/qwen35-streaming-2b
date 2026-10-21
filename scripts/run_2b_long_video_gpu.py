"""Opt-in GPU/score stages bound to completed long-video CPU proof."""
import argparse
from pathlib import Path
import sys
import traceback

sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.wshy.online_memory_2b.isolation import configure,output_path
configure()
from experiments.wshy.online_memory_2b.protocol import save


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',required=True,choices=('gpu','score'))
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args()
    run=output_path(args.run)
    spec=run/'gpu_protocol.json'
    try:
        if args.stage=='gpu':
            from experiments.wshy.long_video_gpu_2b.worker import gpu
            gpu(run,spec)
        else:
            from experiments.wshy.long_video_gpu_2b.scoring import score
            score(run,spec)
    except BaseException:
        failure=run/('long_video_'+args.stage+'_failure.json')
        if run.is_dir() and not failure.exists():
            save(failure,dict(status='failed',traceback=traceback.format_exc(),automatic_retry=False))
        raise


if __name__=='__main__':
    main()
