"""Post-run explanatory diagnostics; never replaces preregistered strict scores."""
from collections import defaultdict
import json
import re
import statistics

from inventory_local import OUT


def main():
    def read(n):return json.loads((OUT/n).read_text(encoding='utf-8'))
    rows={q['id']:q for q in read('stream_requests34.json')}
    labels={q['id']:q for q in read('stream_labels34_LOCAL_ONLY.json')}
    scored=read('strict_results.json');manual={q['id']:q for q in read('manual6_delayed_questions_LOCAL_ONLY.json')}
    sources={q['source_path']:q for q in read('stream_sources6.json')}
    context=defaultdict(list);updates=defaultdict(float);producer=defaultdict(lambda:defaultdict(float))
    bank={};nodes_at={};observations={};format_rows=[]
    for stage in ('native','memory'):
        path=OUT/'gpu_217776'/stage/'events.jsonl'
        for line in path.read_text(encoding='utf-8').splitlines():
            item=json.loads(line);arm=item['arm'];source=item['source_path'];e=item['event']
            if e['kind']=='pair':
                state=item['state'];updates[arm]+=state.get('seconds',0.)
                if state.get('bank') is not None:bank[arm,source]=state['bank']['details']['nodes']
            elif e['kind']=='query':
                ident=e['query_id'];a=item['answer'];context[arm].append(a)
                nodes_at[arm,ident]=bank.get((arm,source),[])
                if 'observation_state' in item:observations[arm,ident]=item['observation_state']
                else:assert arm=='text_only','missing observation state on visual arm'
            elif e['kind']=='end':
                for k,v in item['producer'].items():
                    if k.endswith('_seconds'):producer[arm][k]+=v
    diagnostics={}
    for arm,values in context.items():
        diagnostics[arm]=dict(questions=len(values),first_generated_median_s=statistics.median(v['first_token_seconds'] for v in values),
            full_readout_median_s=statistics.median(v['total_seconds'] for v in values),
            full_readout_sum_s=sum(v['total_seconds'] for v in values),
            input_tokens_sum=sum(v['input_tokens'] for v in values),
            visual_tokens_mean=None if any(v.get('visual_tokens') is None for v in values) else statistics.mean(v['visual_tokens'] for v in values),
            measured_observation_update_sum_s=updates[arm],producer=dict(producer[arm]),
            not_full_end_to_end_latency_or_general_speedup=True)
    for d in scored['details']:
        if d['status']!='format':continue
        match=re.fullmatch(r'\s*([A-E])\.\s*(.*?)\s*',d['text'])
        index=ord(match[1])-65 if match else None
        exact=bool(match and index<len(rows[d['id']]['options']) and
            match[2].strip().casefold()==rows[d['id']]['options'][index].strip().casefold())
        format_rows.append(dict(id=d['id'],arm=d['arm'],text=d['text'],truth=d['truth'],
            explicit_option_letter=index,following_text_exactly_matches_that_option=exact,
            content_correct_diagnostic=exact and index==d['truth'],primary_strict_correct_stays_false=True))
    coverage=[]
    lookup={(d['id'],d['arm']):d for d in scored['details']}
    for ident,q in manual.items():
        spec=sources[q['source_path']];indices={f['frame_id']:i//2 for i,f in enumerate(spec['frames'])}
        pairs=sorted({indices[w['frame_id']] for w in q['witnesses']})
        entry=dict(id=ident,question=q['question'],query_time_s=q['query_time_s'],witness_gap_s=q['witness_gap_s'],
            witness_pair_indices=pairs,answers={},bank_coverage={})
        for arm in ('native','main_only','M0','M2','text_only'):
            d=lookup[ident,arm];entry['answers'][arm]=dict(text=d['text'],correct=d['correct'])
        for arm in ('M0','M2'):
            ns=nodes_at[arm,ident];hit=[]
            for i in pairs:
                containing=[n for n in ns if n['first']<=i<=n['last']]
                hit.append(dict(pair=i,status='missing' if not containing else
                    'exact_raw_pair' if containing[0]['exact_pair'] else 'compressed_provenance_only',
                    source_destination_present=False if not containing else i in containing[0]['surviving_source_pair_ids']))
            entry['bank_coverage'][arm]=dict(nodes=ns,witness_pairs=hit)
        entry['main_state_at_query']=observations['main_only',ident]
        entry['necessity_proven']=False
        coverage.append(entry)
    supported={i for i,r in labels.items() if r['verdict']=='supported'}
    content_supported={arm:sum(d['correct'] for d in scored['details'] if d['arm']==arm and d['id'] in supported)+
        sum(f['content_correct_diagnostic'] for f in format_rows if f['arm']==arm and f['id'] in supported)
        for arm in context}
    result=dict(readout_timing=diagnostics,format_diagnostics=format_rows,
        supported16_content_diagnostic_NOT_primary=content_supported,manual6_coverage=coverage,
        M0_M2_manual_texts_identical=all(lookup[i,'M0']['text']==lookup[i,'M2']['text'] for i in manual),
        candidate_can_cover_history_but_not_proven_readable=True,
        limitations=['coverage is provenance, not recoverable detail','source-destination survival is not information-necessity proof',
          'format diagnostics inspected after run; primary strict score unchanged','single run/fixed order, not a speed benchmark'])
    with (OUT/'explanatory_diagnostics.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(dict(timing=diagnostics,format_diagnostics=format_rows,content_supported=content_supported,
        manual_coverage=[dict(id=r['id'],witness=r['witness_pair_indices'],
            M0=[h['status'] for h in r['bank_coverage']['M0']['witness_pairs']],
            M2=[h['status'] for h in r['bank_coverage']['M2']['witness_pairs']]) for r in coverage]))


if __name__=='__main__':main()
