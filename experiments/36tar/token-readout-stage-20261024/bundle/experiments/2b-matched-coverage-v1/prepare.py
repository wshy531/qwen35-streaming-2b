"""One-time creation: preserve every query and all P3 per-question outcomes."""
from collections import Counter
from pathlib import Path
import json
import shutil
from common import ROOT, OUT, ARMS, read, save, sha


def main():
    source = ROOT/'runs/2b_compressed_banks_prepare_20261023_01'
    OUT.mkdir(exist_ok=False)
    (OUT/'inputs').mkdir()
    for name in ('groups.json','questions.json','snapshots34.json'):
        shutil.copyfile(source/'inputs'/name, OUT/'inputs'/name)
    shutil.copyfile(source/'labels_LOCAL_ONLY.json', OUT/'labels_LOCAL_ONLY.json')
    old_scores = read(source/'answer_results.json')
    details = {(r['id'],r['arm']):r for r in old_scores['details']}
    answers = {}
    for line in (source/'results-218058/answers.jsonl').read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        answers[row['id'],row['arm']] = row
    ledger = []
    for q in read(OUT/'inputs/questions.json'):
        for policy in ('M0','M2','A06','LS'):
            full, compressed = (answers[q['id'],policy+'_'+mode] for mode in ('full48','change24'))
            f, c = (details[q['id'],policy+'_'+mode]['answer_correct'] for mode in ('full48','change24'))
            ledger.append(dict(id=q['id'], group=q['group'], cohort=q['cohort'], policy=policy,
                question=q['question'], options=q['options'], query_time_s=q['query_time_s'],
                full_correct=f, change_correct=c,
                outcome='correction' if c and not f else 'loss' if f and not c else 'both_correct' if f else 'both_wrong',
                full_text=full['text'], change_text=compressed['text'],
                full_first_s=full['nodes'][0]['begin'], change_first_s=compressed['nodes'][0]['begin'],
                full_nodes=full['nodes'], change_nodes=compressed['nodes'],
                input_token_delta=compressed['input_tokens']-full['input_tokens'],
                visual_token_delta=compressed['visual_tokens']-full['visual_tokens']))
    assert len(ledger) == 1212
    save(OUT/'P3_all_question_ledger.json',dict(rows=ledger,
         source_scores_sha256=sha(source/'answer_results.json'),
         all_questions_retained=True, used_to_filter_next_dataset=False))
    references=[]
    mapping={'M0_full48':'full48_8','M0_odd24':'odd24_16','M0_change24':'change24_16'}
    for row in answers.values():
        if row['arm'] in mapping:
            references.append(dict(id=row['id'], arm=mapping[row['arm']], generated_ids=row['generated_ids'],
                input_ids_sha256=row['input_ids_sha256'], node_digest=row['node_digest']))
    assert len(references)==909
    save(OUT/'historical_references_LOCAL_ONLY.json',dict(records=references,
        source_SHA256=sha(source/'results-218058/answers.jsonl')))
    save(OUT/'inputs/protocol.json',dict(arms=list(ARMS),questions=303,answers_max=1515,
         generated_tokens_max=12120,visual_encodings=3981,pretrained_loads=1,
         model_revision='15852e8c16360a2fea060d615a32b45270f8a8fc',dtype='FP32',
         max_new_tokens=8,context_cap=2048,visual_budget_max=384,max_nodes=16,
         matched_frame_pairs=8,matched_compressed_visual_budget=192,expanded_pairs=16,
         selector='frozen C2/P3 change24, previous complete pair, cold only once per source',
         selected_original_values_and_coordinates=True,no_forest_merges=True,
         no_main_language_cache=True,no_full_history_archive=True,
         primary_dataset='opened269; long supported16/manual6 separate; all303 retained',
         contrasts=['odd24_8 vs full48_8','change24_8 vs full48_8','change24_8 vs odd24_8',
                    'odd24_16 vs odd24_8','change24_16 vs change24_8'],
         stop='one frozen five-arm comparison; no ratio, anchor or prompt sweep',
         limitations=['fixed coverage does not fix language sequence length or all position offsets',
                      'M0 isolates pair selection; cannot diagnose all LS/M2 fusion losses',
                      'shared quality timings are not isolated end-to-end speed'],
         input_sha256={n:sha(OUT/'inputs'/n) for n in ('groups.json','questions.json','snapshots34.json')}))
    # Exact copies with explicit local adaptations; old experiment remains untouched.
    old=ROOT/'experiments/2b-compressed-banks-v1'
    here=Path(__file__).parent
    session=(old/'session.py').read_text(encoding='utf-8')
    session=session.replace('compressed-banks-qa','matched-coverage-qa').replace('nine_banks','matched_windows').replace('nine fixed banks required','five fixed windows required')
    (here/'session.py').write_text(session,encoding='utf-8')
    worker=(old/'evaluate.py').read_text(encoding='utf-8')
    worker=worker.replace('One locked nine-bank comparison','One locked matched-coverage comparison')
    worker=worker.replace('== 2727','== 1515').replace('<= 21816','<= 12120')
    worker=worker.replace('nine_banks','matched_windows').replace('< 3300','< 2400').replace('< 96*2**20','< 64*2**20')
    (here/'evaluate.py').write_text(worker,encoding='utf-8')
    scorer=(old/'analyze.py').read_text(encoding='utf-8')
    scorer=scorer.replace("tuple((p+'_change24',p+'_full48') for p in ('M0','M2','A06','LS'))+(('M0_change24','M0_odd24'),)",
        "(('odd24_8','full48_8'),('change24_8','full48_8'),('change24_8','odd24_8'),('odd24_16','odd24_8'),('change24_16','change24_8'))")
    scorer=scorer.replace('expected_answers=2727','expected_answers=1515').replace('expected=2727','expected=1515')
    (here/'analyze.py').write_text(scorer,encoding='utf-8')
    print(dict(status='prepared',arms=ARMS,questions=303,ledger=len(ledger),references=len(references)))


if __name__=='__main__':
    main()
