"""Copy only fixed metadata and unchanged readout source; no media copying."""
import shutil
from pathlib import Path
from common import ROOT, OUT, ARMS, read, save, sha

source = ROOT/'runs/2b_standalone_cost_prepare_20261023_01'
(OUT/'inputs').mkdir(exist_ok=False)
for name in ('groups.json','questions.json','snapshots34.json'):
    shutil.copyfile(source/'inputs'/name, OUT/'inputs'/name)
shutil.copyfile(source/'labels_LOCAL_ONLY.json', OUT/'labels_LOCAL_ONLY.json')
here = Path(__file__).parent
assert not (here/'solo_session.py').exists() and not (here/'io_source.py').exists()
shutil.copyfile(ROOT/'experiments/2b-standalone-cost-v1/session.py', here/'solo_session.py')
shutil.copyfile(ROOT/'experiments/video-input-cpu-v1/source.py', here/'io_source.py')
rows, groups = read(OUT/'inputs/questions.json'), read(OUT/'inputs/groups.json')
assert len(rows) == 303 and len(groups) == 112
pairs = sum(len(g['frames'])//2 + (len(g['frames'])%2 if g['cohort'] != 'stream34' else 0) for g in groups)
assert pairs == 3981
save(OUT/'inputs/protocol.json', dict(arms=list(ARMS), questions=303, sources=112,
    formal_answers=606, warmup_answers=2, answers_max=608, generated_tokens_max=4864,
    formal_visual_calls=pairs*2, warmup_visual_calls=16, numerical_acceptance_visual_calls=4,
    total_visual_calls=7982, pretrained_loads=1, max_new_tokens=8, context_tokens=2048,
    output_visual_tokens_per_pair=48, memory_visual_tokens=384, recent_pairs=8,
    update_ratio=0.25, refresh_interval=2, shared_first_layer_selection=True,
    decoder='auto4', partial_outputs_approximate=True, reference_FP32_bytes=75497472,
    no_cuda_graph=True, no_main_language_cache=True, no_full_feature_archive=True,
    model_revision='15852e8c16360a2fea060d615a32b45270f8a8fc',
    numerical_acceptance=dict(atol=1e-4,rtol=1e-4),
    stop='one fixed first-version comparison; no ratio/interval sweep',
    quality_gate='primary269 and supported_long16 each must not lose net correct answers; paired losses retained',
    speed_gate='candidate total visual seconds must be lower on all sources and long6 aggregates; otherwise no default upgrade',
    timing_scope='shared sequential process and source; descriptive visual timing, not standalone total/edge/1x speed',
    readout_copied_from=str((ROOT/'experiments/2b-standalone-cost-v1/session.py').relative_to(ROOT)),
    input_adapter_copied_from='experiments/video-input-cpu-v1/source.py',
    inputs_SHA256={n:sha(OUT/'inputs'/n) for n in ('groups.json','questions.json','snapshots34.json')}))
old = read(source/'historical_references_LOCAL_ONLY.json')['records']
old = [r for r in old if r['arm']=='full48_8']
assert len(old) == 303
save(OUT/'historical_references_LOCAL_ONLY.json', dict(records=old, copied_from='218063 references via standalone cost'))
print(dict(status='prepared',sources=112,questions=303,formal_answers=606,visual_calls_max=7982))
