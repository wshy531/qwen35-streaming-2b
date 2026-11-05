"""Prepare quality-comparison worker; isolated speed benchmark remains separate."""
from pathlib import Path
import json,hashlib
HERE=Path(__file__).parent;ROOT=HERE.parents[1];OUT=ROOT/'runs/2b_compressed_banks_prepare_20261023_01'
old=ROOT/'experiments/2b-ls-depth-v1'
s=(old/'evaluate.py').read_text(encoding='utf-8').replace('six-bank','nine-bank').replace('six_banks','nine_banks')
s=s.replace('count == 1818 and generated <= 14544','count == 2727 and generated <= 21816')
s=s.replace('else: session.append_pair(pending,packet); pending = None; guard()',
'''else:
                            state=session.append_pair(pending,packet)
                            trace=state['selection_trace']
                            events.write(json.dumps(dict(kind='selection',group=group['id'],
                                trace={k:trace[k] for k in ('pair_index','begin_s','end_s','cold_start','selected_indices','temporal_scores','starts_event','adjacent_distance')}))+'\\n')
                            pending = None; guard()''')
s=s.replace("if pending is not None: session.snapshot_tail(pending)","""if pending is not None:
                        state=session.snapshot_tail(pending);trace=state['selection_trace']
                        events.write(json.dumps(dict(kind='query_only_selection',group=group['id'],
                            trace={k:trace[k] for k in ('pair_index','begin_s','end_s','cold_start','selected_indices','temporal_scores','starts_event','adjacent_distance')}))+'\\n')""")
s=s.replace("def on_event(item):\n                        if", """def on_event(item):
                        if 'state' in item and 'selection_trace' in item['state']:
                            trace=item['state']['selection_trace']
                            item['state']=dict(pairs=item['state']['pairs'],last_pts=item['state']['last_pts'],
                                selection_trace={k:trace[k] for k in ('pair_index','begin_s','end_s','cold_start','selected_indices','temporal_scores','starts_event','adjacent_distance')})
                        if""")
s=s.replace("update_seconds=session.bank.update_seconds)","update_seconds=session.bank.update_seconds,encoding_timing=session.timing)")
(HERE/'evaluate.py').write_text(s,encoding='utf-8',newline='\n')
s=(old/'run_eval.sh').read_text(encoding='utf-8').replace('2b-ls-depth-v1','2b-compressed-banks-v1').replace('2b_ls_depth_prepare_20261021_01','2b_compressed_banks_prepare_20261023_01').replace('2b-ls-depth','2b-compressed-banks')
(HERE/'run_eval.sh').write_text(s,encoding='utf-8',newline='\n')
s=(old/'analyze.py').read_text(encoding='utf-8')
start=s.index("        for candidate, control in ((")
end=s.index(':\n            paired[',start)
s=s[:start]+"        for candidate, control in tuple((p+'_change24',p+'_full48') for p in ('M0','M2','A06','LS'))+(('M0_change24','M0_odd24'),)"+s[end:]
s=s.replace('expected_answers=1818','expected_answers=2727').replace('expected=1818','expected=2727')
s=s.replace("'one seed reservoir',",'')
(HERE/'analyze.py').write_text(s,encoding='utf-8',newline='\n')
run=OUT/'inputs';run.mkdir(exist_ok=False)
source=ROOT/'runs/2b_temporal24_prepare_20261023_01'
for n in ('groups.json','questions.json','snapshots34.json'):(run/n).write_bytes((source/'inputs'/n).read_bytes())
(OUT/'labels_LOCAL_ONLY.json').write_bytes((source/'labels_LOCAL_ONLY.json').read_bytes())
arms=[p+'_'+c for p in ('M0','M2','A06','LS') for c in ('full48','change24')]+['M0_odd24']
protocol=dict(arms=arms,questions=303,answers_max=2727,generated_tokens_max=21816,visual_encodings=3981,
    visual_budget=384,recent_budget=192,M2_closed_budget=144,M2_reserved_open_budget=48,
    compressed_M2_open_actual_tokens=24,compressed_M2_max_used_tokens=360,
    descriptor='uncompressed full48 mean shared boundary across all arms',
    reader_label='point for singleton pair including spatially selected24; interval for temporal merge',
    event_threshold=0.047536224126815796,selector_previous_bytes_at2048=393216,
    previous_is_not_free_evidence=True,no_main_language_cache=True,no_full_history_archive=True,
    max_nodes=16,context_cap=2048,max_new_tokens=8,
    same_frame_coverage_control='separate next comparison, not included in this main nine-arm job',
    independent_speed='not measured by shared quality job',no_default_upgrade=True,
    stop='one frozen nine-arm result, report all quality/cost and regressions, no memory parameter search',
    model_revision='15852e8c16360a2fea060d615a32b45270f8a8fc',dtype='FP32',
    input_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in run.iterdir()})
(run/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n',encoding='utf-8')
print(dict(status='worker_prepared_no_GPU',arms=len(arms),answers=2727))
