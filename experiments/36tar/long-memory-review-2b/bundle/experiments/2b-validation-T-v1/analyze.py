"""Local fixed-denominator analysis; retains errors and all comparison arms."""
from collections import Counter,defaultdict
import json
from pathlib import Path
import random
import sys

from prepare import ROOT,OUT,read,save,sha

sys.path.insert(0,str(ROOT/'experiments/2b-short-event-v1'))
from scoring import measure

ARMS=['native_original','full_pairwise_reference','M0','M1','M2','B02_cap4','reservoir4','text_only']


def paired(rows,predictions,a,b,bootstrap=True):
    corrections=losses=0
    groups=defaultdict(lambda:[0,0])
    for row in rows:
        uid=row['id']
        left=predictions[uid,a]['content_correct']
        right=predictions[uid,b]['content_correct']
        corrections+=int(right and not left)
        losses+=int(left and not right)
        groups[row['source_group_id']][0]+=int(right)-int(left)
        groups[row['source_group_id']][1]+=1
    value=dict(reference=a,candidate=b,questions=len(rows),source_groups=len(groups),
               corrections=corrections,new_losses=losses,net_correct=corrections-losses)
    if bootstrap and groups:
        choices=list(groups.values())
        rng=random.Random(20261020)
        estimates=[]
        for _ in range(2000):
            sample=[rng.choice(choices) for __ in choices]
            estimates.append(sum(x[0] for x in sample)/sum(x[1] for x in sample))
        estimates.sort()
        value['descriptive95percent_source_bootstrap_difference']=[estimates[49],estimates[1949]]
    return value


def analyze(result):
    rows=read(OUT/'scorer_rows_LOCAL_ONLY.json')
    assert len(rows)==len({q['id'] for q in rows})==98
    keys={q['id']:q for q in rows}
    raw=[]
    for stage in ['native','memory']:
        completion=read(result/stage/'completed.json')
        assert completion['status']=='complete'
        lines=[json.loads(line) for line in (result/stage/'predictions.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        assert len(lines)==completion['records']==(98 if stage=='native' else 686)
        raw+=lines
    expected={(uid,arm) for uid in keys for arm in ARMS}
    assert len(raw)==len({(p['id'],p['arm']) for p in raw})==784
    assert {(p['id'],p['arm']) for p in raw}==expected
    predictions={}
    for p in raw:
        q=keys[p['id']]['original']
        score=measure(p['text'] if p['status']=='ok' else '',len(q['options']),q['answer_index'])
        predictions[p['id'],p['arm']]=dict(p,**score)
    subsets=dict(all=rows,FT=[q for q in rows if q['stratum'] in ['F','T']],
                 C=[q for q in rows if q['stratum']=='C'],
                 early_role=[q for q in rows if q['selection_role']=='early_brief_observation_PROXY'],
                 recent_role=[q for q in rows if q['selection_role']=='recent_observation_PROXY'],
                 no_late_same_action=[q for q in rows if q['memory_character']=='past_not_repeated_in_sample'])
    tables={}
    for name,subset in subsets.items():
        tables[name]={arm:dict(questions=len(subset),
            content_correct=sum(predictions[q['id'],arm]['content_correct'] for q in subset),
            strict_correct=sum(predictions[q['id'],arm]['strict_correct'] for q in subset),
            format_compliant=sum(predictions[q['id'],arm]['format_compliant'] for q in subset),
            status_counts=dict(Counter(predictions[q['id'],arm]['status'] for q in subset))) for arm in ARMS}
    comparisons={name:[paired(subset,predictions,a,b) for a,b in [
        ('M0','M2'),('M2','B02_cap4'),('M0','reservoir4')]] for name,subset in subsets.items()}
    matching=0
    for uid in keys:
        native=predictions[uid,'native_original'];full=predictions[uid,'full_pairwise_reference']
        assert native['input_frames_sha256']==full['input_frames_sha256'], 'actual input frame mismatch'
        matching+=int(native['status']==full['status']=='ok' and native['generated_ids']==full['generated_ids'])
    save(result/'analysis.json',dict(status='fixed_all784_records_analyzed',tables=tables,
        comparisons=comparisons,native_full_generated_ID_matches=matching,native_full_total=98,
        raw_questions=119,content_supported=98,other21_retained=True,independent_second_review=False,
        source_sha256=sha(__file__),scorer_rows_sha256=sha(OUT/'scorer_rows_LOCAL_ONLY.json'),
        prospective_analysis_protocol_sha256=sha(OUT/'analysis_protocol.json'),
        limitations=['source-ID groups not proven independent filming','single seed reservoir',
          'C only7 questions,180s source only1 C question','no late same action does not prove necessary old memory',
          'source-bootstrap intervals descriptive, no power or multiple-test significance guarantee',
          'fixed arm order and single pass timing not a speed benchmark']))
    save(result/'per_question_scores.json',list(predictions.values()))
    print(json.dumps(dict(all=tables['all'],FT=tables['FT'],native_full_matches=matching),ensure_ascii=False))


if __name__=='__main__':
    analyze(Path(sys.argv[1]))
