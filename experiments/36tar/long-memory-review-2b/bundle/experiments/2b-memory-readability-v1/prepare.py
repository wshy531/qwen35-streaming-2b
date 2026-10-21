"""One fixed diagnostic: six existing questions, four conditions + two references."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'runs/2b_long_stream_prepare_20261020_01'
OUT=ROOT/'runs/2b_memory_readability_prepare_20261020_01'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(name,value):
    with (OUT/name).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2)


def main():
    manual={q['id']:q for q in read(OLD/'manual6_delayed_questions_LOCAL_ONLY.json')}
    local={s['source_path']:s for s in read(OLD/'stream_sources6.json')}
    remote={s['source_path']:s for s in read(OLD/'runtime_inputs_v2/sources.json')}
    snapshots={s['query_id']:s for s in read(OLD/'stream_query_snapshots34.json')}
    bank={};rows=[];labels=[];references=[];compressions=[]
    for line in (OLD/'gpu_217776/memory/events.jsonl').read_text(encoding='utf-8').splitlines():
        e=json.loads(line)
        if e['arm']!='M2':continue
        if e['event']['kind']=='pair':bank[e['source_path']]=e['state']['bank']['details']['nodes']
        if e['event']['kind']!='query' or e['event']['query_id'] not in manual:continue
        ident=e['event']['query_id'];q=manual[ident];s=local[q['source_path']]
        n=snapshots[ident]['committed_pairs'];nodes=bank[q['source_path']]
        indices={f['frame_id']:i//2 for i,f in enumerate(s['frames'])}
        witnesses=sorted({indices[w['frame_id']] for w in q['witnesses']})
        chosen=[i for i,node in enumerate(nodes) if any(node['first']<=p<=node['last'] for p in witnesses)]
        assert chosen and all(not nodes[i]['exact_pair'] for i in chosen)
        recent=[i for i,node in enumerate(nodes) if node['first']>=n-4]
        assert len(recent)==4 and all(nodes[i]['exact_pair'] for i in recent)
        assert not set(chosen)&set(recent)
        ranges=[[nodes[i]['first'],nodes[i]['last']] for i in chosen]
        raw=[i for a,b in ranges for i in range(a,b+1)]
        recent_pairs=[nodes[i]['first'] for i in recent]
        assert len(raw)==len(set(raw)) and max(raw)<min(recent_pairs)
        rows.append(dict(id=ident,source_path=q['source_path'],query_time_s=q['query_time_s'],
            question=q['question'],options=q['options'],
            media_path=remote[q['source_path']]['media_path'],media_sha256=s['media_sha256'],
            committed_frames=s['frames'][:2*n],committed_frames_sha256=snapshots[ident]['committed_frames_sha256'],
            expected_bank_nodes=nodes,selected_node_indices=chosen,selected_source_ranges=ranges,
            raw_selected_pair_indices=raw,recent_node_indices=recent,recent_pair_indices=recent_pairs,
            total_prefix_pairs=n,local_media_path=s['media_path']))
        labels.append(dict(id=ident,answer_index=q['answer_index'],origin=q['annotation_origin']))
        references.append(dict(id=ident,arm='F_bank_reproduction',generated_ids=e['answer']['generated_ids'],text=e['answer']['text']))
        compressions.append(dict(id=ident,raw_pairs=len(raw),raw_visual_tokens=48*len(raw),
            compressed_visual_tokens=sum(nodes[i]['tokens'] for i in chosen),
            ratio=48*len(raw)/sum(nodes[i]['tokens'] for i in chosen)))
    assert len(rows)==6 and set(q['id'] for q in rows)==set(manual)
    save('requests6_LOCAL.json',rows)
    save('labels6_LOCAL_ONLY.json',labels)
    save('bank_reproduction_reference_LOCAL_ONLY.json',references)
    save('selected_compression_ratios.json',compressions)
    save('diagnostic_plan.json',dict(status='fixed_single_round_pre_GPU',cases=6,
        primary_conditions=['A_raw_old','B_compressed_old','C_raw_old_recent','D_compressed_old_recent'],
        references=['E_full_pairwise','F_bank_reproduction'],answers_max=36,generated_tokens_max=288,
        question_text_and_labels_unchanged=True,
        node_selection='oracle source overlap with pre-output human witnesses; not model-output-conditioned or automatic retrieval',
        A_B_same_old_source_ranges=True,C_D_same_recent4_pairs=True,
        E_reason='distinguish full native protocol from uncompressed pairwise readout',
        F_reason='check replayed bank gives same six outputs as217776',
        hypothesis='test compressed old readout loss versus interference from recent or other old nodes',
        diagnostic_raw_history_is_NOT_bounded_candidate=True,
        snapshot_features_are_limited_diagnostic_outputs_NOT_runtime_archive=True,
        context_guard=32768,visual_guard=16384,node_guard=256,suffix_guard=128,max_new_tokens=8,
        prefill_chunk=512,training=False,download=False,GPU_authorized=False,
        stop='one fixed36-answer run, no prompt/threshold/node sweep; classify outcomes and stop',
        source_locks={str(p.relative_to(ROOT)):sha(p) for p in [
            OLD/'manual6_delayed_questions_LOCAL_ONLY.json',OLD/'stream_labels34_LOCAL_ONLY.json',
            OLD/'gpu_217776/memory/events.jsonl',OLD/'stream_query_snapshots34.json']}))
    print(compressions)


if __name__=='__main__':main()
