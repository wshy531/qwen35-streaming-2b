"""Validate and append an explicitly observed full-prefix content review."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/early25_full_prefix_review_20261020_01'
def main():
    item=json.load(sys.stdin);traces=json.loads((OUT/'image_receipts.json').read_text(encoding='utf-8'));t=next(x for x in traces if x['source_path']==item['source_path'])
    path=OUT/'source_reviews.jsonl';old=[json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []
    if any(x['source_path']==t['source_path'] for x in old):raise ValueError('already reviewed; no overwrite')
    if not item['all_pages_viewed'] or {x['id'] for x in item['questions']}!={x['id'] for x in t['questions']}:raise ValueError('incomplete source review')
    for p in t['pages']:
        if hashlib.sha256((ROOT/p['path']).read_bytes()).hexdigest()!=p['sha256']:raise ValueError('page drift')
    for q in item['questions']:
        if q['content_verdict'] not in ['supported','ambiguous'] or q['memory_character'] not in ['past_with_late_cue','past_not_repeated_in_sample','temporal_relation_ambiguous','answer_not_visible']:raise ValueError('review label')
        q['witnesses']=[]
        for target in q.pop('witness_targets_s'):
            frame=min(t['frames'],key=lambda f:abs(f['pts_s']-target))
            if abs(frame['pts_s']-target)>.1:raise ValueError('not a real sampled witness')
            q['witnesses'].append(frame)
        q.update(original=next(x for x in t['questions'] if x['id']==q['id']),old_evidence_necessity_proven=False)
    item.update(pages=t['pages'],sample_frames=len(t['frames']),reviewer='single_assistant_full_fixed_1FPS_prefix',independent_human_review=False,model_answers_used_for_annotation=False,sampled_absence_not_continuous_absence=True,original171_evaluation_unchanged=True)
    with path.open('a',encoding='utf-8') as f:f.write(json.dumps(item,ensure_ascii=False)+'\n')
    total=sum(len(x['questions']) for x in [*old,item])
    with (ROOT/'实验总纲.md').open('a',encoding='utf-8') as f:f.write(f'\n1.41 D25全前缀审核：{item["source_path"]}的{len(t["pages"])}页全部已看，累计{len(old)+1}/20源、{total}/25题；原171评分不变，未称旧证据必要。\n')
    print(json.dumps(dict(reviewed_sources=len(old)+1,reviewed_questions=total)))
if __name__=='__main__':main()
