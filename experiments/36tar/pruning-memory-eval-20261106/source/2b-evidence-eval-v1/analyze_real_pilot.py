"""Audit actual judge outputs; method-level summaries are exploratory only."""
import collections,json,re
from pathlib import Path
R=Path(__file__).resolve().parents[2];P=R/'runs/evidence_eval_prepare_20261102_01';RUN=R/'runs/evidence-real-judge-20261102-01'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
cases={c['opaque_id']:c for c in read(P/'real_pilot_LOCAL_ONLY.json')};judgments=read(RUN/'judgments.json')
assert len(cases)==len(judgments)==18 and {r['opaque_id'] for r in judgments}==set(cases)
materials=read(P/'real_pilot_material_ids.json') if (P/'real_pilot_material_ids.json').exists() else None
summary={};issues=[];rows=[]
for row in judgments:
    c=cases[row['opaque_id']];response=c['response'];claims=row['fact'].get('claims')
    if not isinstance(claims,list):issues.append(dict(id=row['opaque_id'],issue='claims_unparsed_needs_review'));continue
    allowed=set(materials[c['source_id']]) if materials else None
    text=(response['evidence']+'\n'+response['support']).strip()
    compact=lambda x:re.sub(r'\s+',' ',x).strip().casefold()
    for claim in claims:
        quote=claim.get('quote','')
        if not quote or compact(quote) not in compact(text):issues.append(dict(id=row['opaque_id'],issue='quote_not_exact_excerpt',quote=quote))
        kind=claim.get('kind');verdict=claim.get('visual_verdict')
        if kind!='observation' and verdict is not None:issues.append(dict(id=row['opaque_id'],issue='inference_has_visual_verdict',quote=quote))
        refs=claim.get('references',[])
        if not isinstance(refs,list):issues.append(dict(id=row['opaque_id'],issue='reference_list_needs_review'));continue
        for ref in refs:
            match=re.fullmatch(r'(?:Frame\s+)?([0-9]+)',str(ref).strip())
            if match is None or (allowed is not None and int(match.group(1)) not in allowed):
                issues.append(dict(id=row['opaque_id'],issue='citation_needs_review',reference=ref))
    rows.append(dict(opaque_id=row['opaque_id'],source_id=c['source_id'],question=c['question'],arm=response['arm'],
        support=row['support'].get('sufficiency'),key_visual_error=row['support'].get('key_visual_error'),
        observations=dict(collections.Counter(x.get('visual_verdict') for x in claims if x.get('kind')=='observation')),
        inferences=sum(x.get('kind')=='inference' for x in claims),judge_reason=row['support'].get('reason')))
for arm in ('full','fixed12','heuristic11'):
    rr=[r for r in rows if r['arm']==arm]
    summary[arm]=dict(explanations=len(rr),support=dict(collections.Counter(r['support'] for r in rr)),
        key_visual_error=sum(r['key_visual_error'] is True for r in rr),processing_unknown=sum(r['key_visual_error'] not in (True,False) for r in rr))
out=dict(status='judge_estimates_not_gold',methods=summary,per_explanation=rows,processing_issues=issues,
    source_count=6,questions=6,explanations=18,official_answer_labels_supplied_to_judge=False,
    limitations=['same six questions across three methods, not 18 independent tasks',
        'claim counts depend on verbosity and judge decomposition; not a pooled method accuracy',
        'manual citation and semantic review required; inference/fact conventions may vary'])
(RUN/'analysis.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(methods=summary,processing_issues=issues),ensure_ascii=False))
