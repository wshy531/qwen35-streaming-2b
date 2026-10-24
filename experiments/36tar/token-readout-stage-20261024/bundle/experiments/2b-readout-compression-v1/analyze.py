"""Answer correctness primary, format independent; never drop a failed answer."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from common import OUT, ARMS, read, save, sha

STRICT = re.compile(r'\s*([A-Ea-e])\s*[.!]?\s*')
LEADING = re.compile(r'^\s*([A-Ea-e])(?:\b|[.!)\s])')
EXPLICIT = re.compile(r'^(?:the\s+)?(?:answer|option|choice)(?:\s+is)?\s*[:：]?\s*([A-E])(?:\b|[.)])', re.I)


def norm(text):
    return re.sub(r'[^\w]+',' ',text.lower()).strip()


def parse(text, options, *, status='ok'):
    text = text if isinstance(text,str) else ''
    compliant = STRICT.fullmatch(text) is not None
    if status != 'ok':
        return dict(choice=None,format_compliant=compliant,unjudgeable=True,reason='execution_'+status)
    match = LEADING.match(text) or EXPLICIT.match(text)
    choice = ord(match[1].upper())-65 if match else None
    reason = None
    if choice is not None and not 0 <= choice < len(options):
        choice, reason = None, 'out_of_range_choice'
    if match and choice is not None:
        tail = text[match.end():].strip(' .:：)!,')
        exact = [i for i,o in enumerate(options) if norm(tail) and norm(tail) == norm(o)]
        if len(exact) == 1 and exact[0] != choice:
            choice, reason = None, 'letter_and_answer_conflict'
        elif re.search(r'\b[A-E]\s*[.)、]',tail) or re.match(r'^(?:or|and|/|、)\s*[A-E]\b',tail,re.I):
            choice, reason = None, 'multiple_explicit_choices'
    if choice is None and reason is None:
        exact = [i for i,o in enumerate(options) if norm(text) and norm(text) == norm(o)]
        if len(exact) == 1: choice = exact[0]
        else: reason = 'no_unambiguous_choice'
    return dict(choice=choice,format_compliant=compliant,unjudgeable=choice is None,reason=reason)


def analyze(result, destination):
    rows = {q['id']:q for q in read(OUT/'inputs/questions.json')}
    labels = {q['id']:q for q in read(OUT/'labels_LOCAL_ONLY.json')}
    predictions = {}
    path = result/'answers.jsonl'
    if path.is_file():
        for line in path.read_text(encoding='utf-8').splitlines():
            p = json.loads(line); key = p['id'],p['arm']
            assert key[0] in rows and key[1] in ARMS and key not in predictions
            predictions[key] = p
    details = []
    for uid,q in rows.items():
        for arm in ARMS:
            p = predictions.get((uid,arm))
            parsed = parse('' if p is None else p.get('text'),q['options'],status='missing' if p is None else p.get('status','ok'))
            details.append(dict(id=uid,group=q['group'],cohort=q['cohort'],arm=arm,
                truth=labels[uid]['answer_index'],text=None if p is None else p.get('text'),
                answer_correct=parsed['choice'] == labels[uid]['answer_index'],**parsed))
    lookup = {(r['id'],r['arm']):r for r in details}
    subsets = {c:[uid for uid,q in rows.items() if q['cohort'] == c] for c in ('natural171','T98')}
    subsets['primary269'] = subsets['natural171']+subsets['T98']
    subsets['stream_original28'] = [uid for uid,q in rows.items() if q['cohort'] == 'stream34' and uid.startswith('ovo:')]
    subsets['stream_supported16'] = [uid for uid in subsets['stream_original28'] if labels[uid]['verdict'] == 'supported']
    subsets['stream_manual6'] = [uid for uid,q in rows.items() if q['cohort'] == 'stream34' and not uid.startswith('ovo:')]
    for cohort in ('natural171','T98','primary269'):
        for stratum in ('F','T','C'):
            subsets[cohort+'_'+stratum] = [uid for uid in subsets[cohort] if labels[uid]['stratum'] == stratum]
    def scores(ids, arm):
        rs = [lookup[uid,arm] for uid in ids]; groups = defaultdict(list)
        for r in rs: groups[r['group']].append(int(r['answer_correct']))
        return dict(correct=sum(r['answer_correct'] for r in rs),denominator=len(rs),
            accuracy=sum(r['answer_correct'] for r in rs)/len(rs) if rs else None,
            format_compliant=sum(r['format_compliant'] for r in rs),
            unjudgeable=sum(r['unjudgeable'] for r in rs),
            reasons=dict(Counter(r['reason'] for r in rs if r['reason'])),
            source_protocol_macro=sum(sum(v)/len(v) for v in groups.values())/len(groups) if groups else None)
    tables = {}
    for name,ids in subsets.items():
        paired = {}
        for candidate, control in tuple((p+'_read24',p+'_full48') for p in ('M2','A06','LS')):
            paired[candidate+'_vs_'+control] = dict(
                corrections=[uid for uid in ids if lookup[uid,candidate]['answer_correct'] and not lookup[uid,control]['answer_correct']],
                new_losses=[uid for uid in ids if lookup[uid,control]['answer_correct'] and not lookup[uid,candidate]['answer_correct']])
        tables[name] = dict(scores={a:scores(ids,a) for a in ARMS},paired=paired)
    historical_path = OUT/'historical_references_LOCAL_ONLY.json'
    replay = None
    if historical_path.exists():
        historical = read(historical_path)['records']
        replay = dict(expected=len(historical),
            generated_ID_mismatches=[dict(id=r['id'],arm=r['arm']) for r in historical
                if predictions.get((r['id'],r['arm']),{}).get('generated_ids') != r['generated_ids']],
            input_ID_mismatches=[dict(id=r['id'],arm=r['arm']) for r in historical
                if r.get('input_ids_sha256') and predictions.get((r['id'],r['arm']),{}).get('input_ids_sha256') != r['input_ids_sha256']],
            node_mismatches=[dict(id=r['id'],arm=r['arm']) for r in historical
                if r.get('node_digest') and predictions.get((r['id'],r['arm']),{}).get('node_digest') != r['node_digest']],
            SHA256=sha(historical_path),note='report all mismatches, no outcome-based removal or silent rerun')
    cost = {}
    for arm in ARMS:
        ps = [p for p in predictions.values() if p['arm'] == arm]
        means = {}
        for key in ('input_tokens','visual_tokens','QA_total_seconds','prefill_seconds','decode_seconds','QA_cache_bytes'):
            values = [p[key] for p in ps if key in p]
            means[key] = sum(values)/len(values) if values else None
        cost[arm] = dict(mean=means,query_max_tree_depth=max((n['depth'] for p in ps for n in p.get('nodes',[])),default=None))
    save(destination,dict(metric='answer_correctness_primary_format_separate',expected_answers=1818,
        actual_answers=len(predictions),tables=tables,details=details,
        historical_replay=replay,descriptive_shared_process_cost=cost,
        labels_sha256=sha(OUT/'labels_LOCAL_ONLY.json'),protocol_sha256=sha(OUT/'inputs/protocol.json'),
        limitations=['opened development and regression only','34 long queries share6sources',
            'ambiguity and sampling limitations unchanged','no independent content second review',
            'format not multiplied into accuracy','bank-only costs not comparable to observation-main runs']))
    print(dict(answers=len(predictions),expected=1818,tables=len(tables)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--results',type=Path,required=True); parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args(); analyze(args.results,args.out)
