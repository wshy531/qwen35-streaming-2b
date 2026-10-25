"""S-EMBER 102 official questions: media, prepare, CPU, GPU, score."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.wshy.online_memory_2b.isolation import configure

configure()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True,
                        choices=('media', 'prepare', 'cpu', 'gpu', 'score'))
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--model-dir', type=Path)
    args = parser.parse_args()
    if args.stage == 'media':
        from experiments.wshy.sember_102_2b.sampling import media
        media()
    elif args.stage == 'prepare':
        if args.model_dir is None:
            parser.error('prepare requires --model-dir')
        from experiments.wshy.sember_102_2b.runner import prepare
        prepare(args.out, args.model_dir)
    elif args.stage == 'cpu':
        from experiments.wshy.sember_102_2b.runner import cpu
        cpu(args.out)
    elif args.stage == 'gpu':
        from experiments.wshy.sember_102_2b.gpu import gpu
        gpu(args.out)
    else:
        from experiments.wshy.sember_102_2b.scoring import score
        score(args.out)


if __name__ == '__main__':
    main()
