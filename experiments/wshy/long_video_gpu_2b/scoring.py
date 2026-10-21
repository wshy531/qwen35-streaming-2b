"""Score locked outputs separately by original and historical admission labels."""
from collections import Counter
import importlib.util

from ..online_memory_2b.frozen import PACKAGE
from ..online_memory_2b.protocol import read,rows,save,sha
from ..online_memory_2b.isolation import output_path
from .gates import verify_spec


def scorer():
    path=PACKAGE/'bundle/experiments/2b-short-event-v1/scoring.py'
    spec=importlib.util.spec_from_file_location('_long_gpu_frozen_scorer',path)
    value=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def build(queries,catalog,predictions,measure):
    arms=['M0','M2','M2_A06','LS_shared384_v1','reservoir4','text_only']
    byid={q['id']:q for q in queries}
    labels={r['id']:r for r in catalog}
    expected={(qid,a) for qid in byid for a in arms}
    indexed={(p['id'],p['arm']):p for p in predictions}
    if (len(queries)!=34 or len(byid)!=34 or len(labels)!=len(catalog) or set(labels)!=set(byid)
            or len(predictions)!=204 or set(indexed)!=expected):
        raise ValueError('complete unique 34-question/six-arm grid required')
    groups={name:[] for name in ('original28_raw_labels','historical_supported16','historical_ambiguous11',
                                'historical_cutoff1','manual6_source_declared')}
    details=[]
    for qid,q in byid.items():
        label=labels[qid]
        truth=label['label_index']
        if (type(truth) is not int or not 0<=truth<len(q['options'])
                or label['independent_review']!='pending' or label['confirmed_accuracy_eligible'] is not False):
            raise ValueError('label or content-review state drift')
        if qid.startswith('ovo:'):
            if label['label_origin']!='exactly_matched_official_metadata':
                raise ValueError('original label provenance drift')
            groups['original28_raw_labels'].append(qid)
            name={'supported':'historical_supported16','ambiguous':'historical_ambiguous11',
                  'not_established_at_committed_cutoff':'historical_cutoff1'}.get(label['declared_admission'])
            if name is None:
                raise ValueError('unknown historical admission declaration')
            groups[name].append(qid)
        elif qid.startswith('local_longstream_v1:'):
            if (label['label_origin']!='single_assistant_manual_source_declaration'
                    or label['declared_admission']!='manual_single_reviewer_candidate'):
                raise ValueError('manual label provenance drift')
            groups['manual6_source_declared'].append(qid)
        else:
            raise ValueError('unknown question cohort')
        for arm in arms:
            p=indexed[qid,arm]
            if p['status']!='ok' or not isinstance(p['text'],str):
                raise ValueError('failed or invalid answer')
            details.append(dict(id=qid,arm=arm,source_path=q['source_path'],cohort=q['cohort'],
                declared_admission=label['declared_admission'],label_origin=label['label_origin'],
                independent_review='pending',**measure(p['text'],len(q['options']),truth)))
    if [len(v) for v in groups.values()]!=[28,16,11,1,6]:
        raise ValueError('original/admission/manual scoring denominators changed')
    values={(r['id'],r['arm']):r for r in details}
    tables,paired={},{}
    for name,ids in groups.items():
        tables[name]={}
        paired[name]={}
        for arm in arms:
            table=dict(questions=len(ids))
            for metric in ('strict_correct','content_correct','format_compliant'):
                table[metric]=sum(values[qid,arm][metric] for qid in ids)
            tables[name][arm]=table
        for ref in ('M0','M2','M2_A06','reservoir4'):
            paired[name][ref]={}
            for arm in (a for a in arms if a not in (ref,'text_only')):
                paired[name][ref][arm]={}
                for metric in ('strict_correct','content_correct'):
                    gains=[qid for qid in ids if not values[qid,ref][metric] and values[qid,arm][metric]]
                    losses=[qid for qid in ids if values[qid,ref][metric] and not values[qid,arm][metric]]
                    paired[name][ref][arm][metric]=dict(corrections=gains,new_losses=losses,net=len(gains)-len(losses))
    # Source groups remain separate and retain their declared label status.
    per_source={}
    for source in sorted({q['source_path'] for q in queries}):
        per_source[source]={}
        for name,ids in groups.items():
            selected=[qid for qid in ids if byid[qid]['source_path']==source]
            if selected:
                per_source[source][name]={arm:dict(questions=len(selected),strict_correct=sum(
                    values[qid,arm]['strict_correct'] for qid in selected)) for arm in arms}
    review=[dict(id=q['id'],source_path=q['source_path'],query_time_s=q['query_time_s'],
        question=q['question'],options=q['options'],reviewer=None,review_status='pending',
        sampled_evidence_present=None,later_repetition_or_context_cues=None,
        long_memory_necessary=None,semantic_answer_judgment=None) for q in queries]
    return dict(questions=34,main_answers=204,tables=tables,paired_comparisons=paired,
        per_source=per_source,confirmed_accuracy=None,confirmed_accuracy_denominator=0,
        score_interpretation='frozen strict/leading option parser against labels; no human semantic judgment',
        original_admission='historical source declarations; original content ledger missing',
        manual_labels='single-assistant source declarations; independent review pending',
        independent_holdout=False,human_review='pending',stable_ability_improvement_established=False),details,review


def score(run,spec_path):
    spec,protocol,proof,(queries,sources,snapshots)=verify_spec(run,spec_path)
    run=output_path(run)
    complete=read(run/'gpu/completed.json')
    outputs={name:sha(run/'gpu'/name) for name in ('predictions.jsonl','events.jsonl')}
    if (complete['status']!='complete' or complete['GPU_protocol_sha256']!=sha(spec_path)
            or complete['CPU_preflight_sha256']!=sha(run/'cpu/preflight.json')
            or complete['output_sha256']!=outputs or complete['main_answers']!=204):
        raise ValueError('complete hash-bound GPU output required')
    summary,details,review=build(queries,read(run/'scoring_ONLY.json'),
                               rows(run/'gpu/predictions.jsonl'),scorer().measure)
    summary.update(GPU_protocol_sha256=sha(spec_path),completed_sha256=sha(run/'gpu/completed.json'),
                   raw_output_hashes=outputs,scope='raw label and historical admission strata, never pooled confirmed accuracy')
    out=output_path(run/'score')
    out.mkdir(exist_ok=False)
    save(out/'summary.json',summary)
    save(out/'details.json',details)
    save(out/'review_without_predictions.json',review)
