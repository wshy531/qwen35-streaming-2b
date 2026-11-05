"""CPU-only evidence index. Does not train, score new model outputs or alter runs."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/eval_inventory_20261104_01'
sources = {}
rows = []


def read(name):
    path = ROOT / name
    raw = path.read_bytes()
    sources[name] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def add(panel, method, reading, correct, n, protocol, source, note=''):
    assert isinstance(correct, int) and 0 <= correct <= n
    rows.append(dict(panel=panel, method=method, reading=reading, correct=correct,
                     judgments=n, percent=round(100*correct/n, 2), protocol=protocol,
                     source=source, limitation=note))


s = 'runs/repair-guard-220499/analysis.json'
d = read(s)
for view, methods in d['metrics'].items():
    panel = 'N345' if view.startswith('old_') else 'N271'
    reading = view.split('/')[1]
    for method, val in methods.items():
        add(panel, method, reading, val['correct'], val['answers'], 'option_logits_2orders', s)
s = 'runs/training_order_main_prepare_20261104_02/frozen_baseline_metrics.json'
for key, val in read(s).items():
    if key.endswith('/text'):
        add('N345' if 'old_development' in key else 'N271', 'text_only', 'none',
            val['correct'], val['answers'], 'option_logits_2orders', s)
s = 'runs/history-utility-short-220532/pruning_comparison.json'
d = read(s)
add('N345', 'full108', 'full_short', d['unpruned_correct'], d['judgments'], 'option_logits_2orders', s)
s = 'runs/history-utility-short-220532/LS_controls.json'
for method, val in read(s).items():
    add('N345', 'full108', method, val['correct'], val['answers'], 'option_logits_2orders', s,
        'LS fixed every 3 pairs; recent432 means 4 unpruned pairs')
s = 'runs/short-ls-fill-220545/analysis.json'
for method, val in read(s)['methods'].items():
    add('N345', method, 'LS', val['correct'], val['answers'], 'option_logits_2orders', s,
        'LS864/recent432 fixed every 3 pairs; 54 tokens per pair')
s = 'runs/learned-prune-main-220024/analysis.json'
d = read(s)['quality']['development']['methods']['fixed12']
add('N345', 'fixed12', 'full_short', d['two_rotations_correct'], d['two_rotations_answers'],
    'generated_option_2orders', s, 'Historical generation protocol, not current logit classification')
s = 'runs/memory-prune-panel150-220382/analysis_LOCAL_ONLY.json'
for dataset, val in read(s)['main'].items():
    for arm, metric in val['scores'].items():
        method, reading = arm.split('/')
        add('SB75' if dataset == 'StreamingBench' else 'N75', method, reading,
            metric['correct'], metric['questions'], 'B_explanation_option_1order', s,
            'Adaptive event boundaries; budget864; M0 full108=8 pairs, pruned54=16 pairs')
s = 'runs/memory-prune-panel150-220382/text75_summary_LOCAL_ONLY.json'
d = read(s)
add('SB75', 'text_only', 'none', d['correct'], d['questions'], 'B_explanation_option_1order', s)
s = 'runs/protect-stream-merged-220270-220293/analysis.json'
for method, val in read(s)['overall'].items():
    add('SB150', 'full108', method, val['correct'], val['questions'], 'protection_short_generation', s,
        'Different prompt/output budget from SB75 B explanation experiment')
s = 'runs/memory-followup-220335/analysis_148.json'
for method, val in read(s)['overall'].items():
    add('SB148', 'full108', method, val['correct'], val['n'], 'protection_short_generation', s,
        'Common 148 only; two missing, not full 150')
s = 'runs/midprune-memory-220237/analysis_content.json'
for method, val in read(s)['methods'].items():
    for role in ('current', 'old'):
        v = val[role]
        add('D18_'+role, method.split('/')[0], method.split('/')[1],
            v['option_content_correct'], v['questions'], 'B_explanation_option_1order', s,
            '5 familiar sources, 12 current and 6 old questions; paired developmental diagnosis')
s = 'runs/sember_main_prepare_20261025_01/formal_analysis.json'
for group, methods in read(s)['group_scores'].items():
    for method, val in methods.items():
        add('SE478_'+group, method, 'encoded_in_method', val['correct'], val['total'],
            'generated_option_16tokens_1order', s,
            'Old adaptive events; ge120/ge300 use author evidence-end proxy, not proven necessity')
s = 'runs/sember_text_control_prepare_20261025_01/text_comparison_analysis.json'
for group, val in read(s)['scores'].items():
    if group in ('all', 'ge120s', 'ge300s'):
        add('SE478_'+group, 'text_only', 'none', val['text_correct'], val['total'],
            'generated_option_16tokens_1order', s)

s = 'runs/sember-pruned-ls-220548/analysis.json'
for method, groups in read(s)['scores'].items():
    for group, metric in groups.items():
        add('SE478_pruned_LS_'+group, method, 'LS', metric['correct'], metric['total'],
            'generated_option_16tokens_1order', s, 'Fixed events every 3 pairs; 54 tokens/pair')

for panel_id, source, result_key in [
    ('N271', 'runs/history-utility-fit-220511/completed.json', 'results'),
    ('N345', 'runs/history-utility-short-heads-220532/completed.json', 'bounded_LS_heads')]:
    completed = read(source)
    assert completed['status'] == 'completed'
    for head, report in completed[result_key].items():
        for reading, metric in report['metrics'].items():
            add(panel_id, 'utility_'+head, reading, metric['correct'], metric['answers'],
                'option_logits_2orders', source,
                'Frozen unpruned108 LS864/recent432; 1114 train questions, 20 epochs; heads separate from pruning')
    if panel_id == 'N271':
        metrics = completed[result_key]['visible']['metrics']
        for reading, key in [('recent4pairs','recent'),('LS','all')]:
            add(panel_id, 'full108', reading, metrics[key]['correct'], metrics[key]['answers'],
                'option_logits_2orders', source, 'No pruning; recent432 means 4 pairs, not 8')

panel = read('runs/memory_prune_panel150_prepare_20261103_01/regression150.json')
large = read('runs/evidence_scale_prepare_20261102_01/panel.json')
assert isinstance(large, list)
large_ids = {q['id'] for q in large}
assert all(q['id'] in large_ids for q in panel)
assert sum(q['dataset'] == 'StreamingBench' for q in panel) == 75
keys = [(r['panel'], r['method'], r['reading'], r['protocol']) for r in rows]
assert len(keys) == len(set(keys))
OUT.mkdir(parents=True, exist_ok=True)
(OUT/'results.json').write_text(json.dumps(dict(snapshot_date='2026-11-05',
    scope='Verified pruning-memory comparison families; older experiments indexed in the companion document',
    continuing_status_record='实验总纲.md', rows=rows, source_sha256=sources,
    checks=dict(unique_rows=True, panel150_subset_of_panel300=True,
                no_new_model_inference=True, no_training=True)), ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
lines = ['# 已核验结果明细（2026-11-05静态快照）', '',
         '由 build.py 从已有分析文件提取。不同panel或protocol不得合并排名；持续状态只看实验总纲。', '',
         '| 面板 | 方法 | 读取方式 | 正确/判断 | 正确率 | 评分协议 |',
         '|---|---|---|---:|---:|---|']
for r in rows:
    lines.append(f"| {r['panel']} | {r['method']} | {r['reading']} | {r['correct']}/{r['judgments']} | {r['percent']:.2f}% | {r['protocol']} |")
(OUT/'结果明细.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
print(json.dumps(dict(rows=len(rows), source_files=len(sources), output=str(OUT)), ensure_ascii=False))
