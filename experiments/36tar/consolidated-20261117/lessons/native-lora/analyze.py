import argparse,json,re
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[2];PREP=ROOT/'runs/native_memory_eval_prepare_20261117_01'
def choice(text,n):
    text=text.strip().upper();allowed='ABCDE'[:n]
    if re.fullmatch('['+allowed+']',text):return ord(text)-65
    explicit=re.findall(r'(?:ANSWER|OPTION|CHOICE)\s*(?:IS|:)?\s*\(?(['+allowed+r'])\b',text)
    candidates=explicit if explicit else re.findall(r'\b(['+allowed+r'])\b',text)
    return ord(candidates[0])-65 if len(set(candidates))==1 else None
def main():
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);p.add_argument('--suite',choices=('stream','short'),required=True);a=p.parse_args()
    rows=json.loads((PREP/(a.suite+'.json')).read_text());labels=json.loads((PREP/'labels_LOCAL_ONLY.json').read_text());qs={r['id']:r for r in rows}
    predictions=[json.loads(s) for s in (a.folder/'answers.jsonl').read_text().splitlines()]
    expected={(q,m,r) for q in qs for m in ('base','ordinary','ls') for r in (('recent864','ls864','text','recent432') if m=='base' else ('recent864','ls864','text'))}
    actual={(r['id'],r['model'],r['read']) for r in predictions};assert len(actual)==len(predictions) and actual==expected
    assert json.loads((a.folder/'completed.json').read_text())['status']=='complete'
    table={};scores={};by_type={}
    for r in predictions:
        selected=choice(r['text'],len(qs[r['id']]['options']));method=r['model']+'_'+r['read'];key=(r['id'],method)
        scores[key]=selected==labels[r['id']];t=table.setdefault(method,Counter())
        t['n']+=1;t['content_correct']+=scores[key];t['unparsed']+=selected is None;t['single_letter_format']+=bool(re.fullmatch('[A-E]',r['text'].strip().upper()))
        category=qs[r['id']]['question_type'];ct=by_type.setdefault(category,{}).setdefault(method,Counter())
        ct['n']+=1;ct['content_correct']+=scores[key]
    pairs={}
    for method in table:
        for baseline in ('base_recent864','base_ls864','base_recent432'):
            pairs[method+' vs '+baseline]=dict(gain=sum(scores[q,method] and not scores[q,baseline] for q in qs),loss=sum(not scores[q,method] and scores[q,baseline] for q in qs))
    result=dict(suite=a.suite,questions=len(rows),sources=len({r['source_id'] for r in rows}),table=table,by_official_type=by_type,paired=pairs,
        metric='generated option content accuracy; format reported separately',opened_development=True)
    (a.folder/'analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
