"""Comparison-only source timing audit; holdout semantics are not consulted.

The original OVO questions/query/GT are unchanged. Anticipation is a valid
separate task; future action labels must not be scored as failures of recall.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath

from audit_development_evidence import crosstask_question, normalize, parse_crosstask


def main():
    root=Path(__file__).resolve().parents[2];run=root/'runs/dataset_build_20261018_01'
    metadata=run/'source_metadata'
    rows=json.loads((root/'experiments/2b-memory-eval/data/candidate_manifest.json').read_text(encoding='utf-8'))
    cross=parse_crosstask(root/'runs/evidence_review_20261017_01/crosstask_release.zip')
    coin={key:dict(member='COIN.json/'+key,steps=[dict(action=normalize(s['label']),
        start_s=s['segment'][0],end_s=s['segment'][1]) for s in value['annotation']])
        for key,value in json.loads((root/'runs/development_audit_20261017_02/COIN.json').read_text())['database'].items()}
    cooking={key:dict(member='COOT_copy/youcookii_annotations_trainval.json/'+key,steps=[
        dict(action=normalize(s['sentence']),start_s=s['segment'][0],end_s=s['segment'][1])
        for s in value['annotations']]) for key,value in json.loads(
            (metadata/'youcookii_annotations_trainval_coot_copy.json').read_text())['database'].items()}
    hirest=json.loads((metadata/'hirest_all_data_val.json').read_text())
    entries=[]
    for row in rows:
        if row['split']!='comparison' or row['dataset']!='ovo':continue
        family=row['source_path'].split('/')[0]
        if family in ('cross_task','COIN','youcook2'):
            annotations={'cross_task':cross,'COIN':coin,'youcook2':cooking}[family]
            audit=crosstask_question(row,annotations)
            audit['time_source_provenance']='secondary_COOT_redistribution_original_site_unavailable' if family=='youcook2' else 'author_annotation'
        elif family=='hirest':
            stem=PurePosixPath(row['source_path']).stem
            video,a,b=stem.rsplit('_',2);a,b=int(a),int(b)
            matches=[v for group in hirest.values() for key,v in group.items()
                if PurePosixPath(key).stem==video and v.get('bounds')==[a,b]]
            if len(matches)!=1:raise ValueError('HiREST trim ambiguous')
            local=dict(member='HiREST/all_data_val.json/'+video+f'/trim_{a}_{b}',steps=[
                dict(action=normalize(s['heading']),start_s=s['absolute_bounds'][0]-a,
                    end_s=s['absolute_bounds'][1]-a) for s in matches[0]['steps']])
            audit=crosstask_question(row,{stem:local})
            audit.update(time_source_provenance='author_annotation',original_clip_offset_s=a)
        else:
            audit=dict(status='no_independent_action_timing_source',time_source_provenance=None)
        entries.append(dict(id=row['id'],query_time_s=row['query_time_s'],**audit))
    result=dict(status='comparison_metadata_timing_not_model_accuracy_or_erratum',
        questions=entries,status_counts=dict(Counter(q['status'] for q in entries)),
        original_questions_answers_query_times_unchanged=True,holdout_semantics_reviewed=False,
        local_metadata_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [metadata/'hirest_all_data_val.json',metadata/'youcookii_annotations_trainval_coot_copy.json']})
    (run/'ovo_comparison_temporal_final.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result['status_counts']))


if __name__=='__main__':main()
