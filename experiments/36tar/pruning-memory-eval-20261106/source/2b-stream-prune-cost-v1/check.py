"""CPU regression for timing wrapper and predecessor-restored input equivalence."""
import argparse
import hashlib
import os
from pathlib import Path
from cost_common import ARMS, PREP, ROOT, read, scale, timed_generate, write


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    env = scale.s.runtime()
    torch = env['torch']
    torch.set_num_threads(1)
    torch.manual_seed(20261102)
    processor = env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B', local_files_only=True)
    config = env['tiny_config'](processor)
    config['vision_config']['depth'] = 24
    model = env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](**config, attn_implementation='sdpa')).float().eval().requires_grad_(False)
    builder = scale.s.make_builder(env, processor)
    from transformers.feature_extraction_utils import BatchFeature
    batches = [BatchFeature(dict(pixel_values_videos=torch.randn(432,1536), video_grid_thw=torch.tensor([[1,18,24]]))) for _ in range(10)]
    questions = read(PREP/'panel.json')
    references = {(r['id'],r['arm']):r for r in read(PREP/'reference.json')}
    digest = hashlib.sha256()
    layouts = 0
    checks = []
    with torch.no_grad(), env['fp32_reference']():
        for arm in ARMS:
            selector = scale.s.CoverageScorer()
            full = []
            for index, batch in enumerate(batches):
                features, ids, _ = scale.encode(model, batch, arm, None, None, selector)
                full.append((index,2.*index,2.*index+1,features.clone(),ids.clone()))
            selector = scale.s.CoverageScorer()
            replay = []
            for index in range(1,10):
                features, ids, _ = scale.encode(model, batches[index], arm, None, None, selector)
                if index >= 2:
                    assert torch.equal(features, full[index][3]) and torch.equal(ids, full[index][4])
                    replay.append((index,2.*index,2.*index+1,features,ids))
            old = scale.generate(env, model, builder, replay, questions[0], arm, limit=4)
            new = timed_generate(env, model, builder, full[-8:], questions[0], arm, limit=4)
            for name in old:
                assert old[name] == new[name], (arm, name)
            checks.append(dict(arm=arm, replay_pairs=8, exact_features=True, exact_generation=True, generated_ids=new['generated_ids']))
            for q in questions:
                ref = references[(q['id'],arm)]
                records = [(pi,float(2*pi),float(2*pi+1),full[-1][3],full[-1][4]) for pi in ref['visible_pairs']]
                emb, pos, ids = scale.layout(env, model, builder, records, q, arm)
                assert ids.shape[1]+128 <= 4096
                assert sum(len(r[3]) for r in records) == ref['visual_tokens']
                digest.update(ids.numpy().tobytes())
                digest.update(pos.numpy().tobytes())
                layouts += 1
    assert layouts == 60
    write(args.out, dict(status='passed_random_weights_only', layouts=layouts, layout_sha256=digest.hexdigest(),
                         checks=checks, tiny_answers=6, replay_pair_checks=24, GPU=0, real_weight_regression_pending=True))
    print(dict(status='passed', layouts=layouts, replay_pairs=24, tiny_answers=6))


if __name__ == '__main__':
    main()
