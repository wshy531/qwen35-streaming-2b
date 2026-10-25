"""Prepare locally; CPU/GPU execution requires separately approved Slurm work."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.wshy.online_memory_2b.isolation import configure
configure()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=('plan', 'cpu', 'gpu', 'score'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path)
    args = parser.parse_args()
    from experiments.wshy.sember_readout_diagnostic import runner
    if args.stage == 'cpu':
        if args.model_dir is None:
            parser.error('CPU stage requires --model-dir')
        runner.cpu(args.out, args.model_dir)
    else:
        getattr(runner, args.stage)(args.out)


if __name__ == '__main__':
    main()
