"""Prepare the fixed opened panel for the author's real-task reader; no model execution."""
import ast,collections,hashlib,json,re,math
from pathlib import Path
R=Path(__file__).resolve().parents[2]
P=R/'runs/malow_source_review_20261103_01'
REMOTE='/home/u2025010462/share/czy/swt/my_pape/qwen35-streaming-2b-36tar'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2)

def main():
    src=P/'source/src/malow/video/streaming.py'
    tree=ast.parse(src.read_text(encoding='utf-8'))
    names={'format_options','question_id','interactive_timestamp'}
    definitions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in definitions}==names
    # Execute only three reviewed pure formatting/time/id helpers, not the author's worker.
    scope={'Any':object,'OPTION_LETTERS':tuple('ABCD'),'re':re,'math':math}
    exec(compile(ast.Module(body=definitions,type_ignores=[]),str(src),'exec'),scope)
    qpath=R/'runs/protection_streamingbench_prepare_20261103_01/questions.json'
    questions=read(qpath);assert len(questions)==150
    groups=collections.defaultdict(list)
    for q in questions:groups[q['source_id']].append(q)
    entries=[];mapping={};types=collections.Counter()
    for source,qs in sorted(groups.items()):
        e={'video_path':REMOTE+'/'+qs[0]['media_path'],'questions':[]}
        for q in sorted(qs,key=lambda x:(x['query_time_s'],x['id'])):
            assert len(q['options'])==4
            converted=dict(question=q['question'],options=q['options'],time_stamp=q['original_timestamp'],
                task_type=q['type'],required_ability=q.get('required_ability'),
                _source_entry_index=q['original_annotation_row'],_source_question_index=q['original_question_index'])
            assert scope['interactive_timestamp'](converted['time_stamp'])==q['query_time_s']
            assert scope['format_options'](q['options'])==[f'{chr(65+i)}. {x}' for i,x in enumerate(q['options'])]
            key=scope['question_id']('real',len(entries),len(e['questions']),converted)
            assert key not in mapping
            mapping[key]=dict(project_id=q['id'],source_id=source,media_sha256=q['media_sha256'])
            assert not any(k in converted for k in ('answer','correct_index','correct_letter'))
            e['questions'].append(converted);types[q['type']]+=1
        entries.append(e)
    assert len(entries)==144 and len(mapping)==150
    out=P/'panel';out.mkdir(exist_ok=False)
    save(out/'questions_real.json',entries);save(out/'id_mapping.json',mapping)
    save(out/'CPU_contract.json',dict(passed=True,questions=150,media_files=144,types=types,
        source_sha256=hashlib.sha256(qpath.read_bytes()).hexdigest(),author_source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
        checks=['lossless question/options','exact question timestamp','unique native IDs','no answers in reader file'],
        labels='reuse project LOCAL_ONLY label file in offline scoring; do not use author summary without labels',
        protocol='author baseline only; prompt/sampling differ from project until explicitly aligned',
        instructions='direct worker --task real --data-file questions_real.json; top-level launcher defaults to all four tasks',
        limit='Schema helper check only; no media decoding/model/GPU/weight download'))
    print(json.dumps(dict(passed=True,questions=150,files=144,types=types)))
if __name__=='__main__':main()
