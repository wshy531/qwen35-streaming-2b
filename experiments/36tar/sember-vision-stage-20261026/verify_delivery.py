"""Standard-library integrity, dependency and answer audit. Never imports model/runtime code."""
import argparse
import ast
from collections import Counter,defaultdict
import gzip
import hashlib
import json
from pathlib import Path,PurePosixPath
import re


def verify(base,labels=None,materialize=None):
    base=base.resolve();repo=base.parents[2]
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    manifest=read(base/'manifest.json');assets=read(base/'external_assets.json');external=read(base/'external_inputs.json')
    def data(rel):
        spec=manifest[rel];p=(repo/spec['stored_path']).resolve();assert p.is_relative_to(repo)
        stored=p.read_bytes();assert hashlib.sha256(stored).hexdigest()==spec['sha256'],rel
        raw=gzip.decompress(stored) if spec['gzip'] else stored
        assert len(raw)==spec['original_bytes'] and hashlib.sha256(raw).hexdigest()==spec['original_sha256'],rel
        return raw
    def load(rel):return json.loads(data(rel))
    forbidden={'.mp4','.mkv','.webm','.jpg','.jpeg','.png','.npy','.npz','.pt','.pth','.safetensors','.pem','.key'}
    secret=re.compile(rb'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----|github_pat_[A-Za-z0-9_]{30,}|gh[pousr]_[A-Za-z0-9]{25,}|hf_[A-Za-z0-9]{25,}')
    python=0
    for rel,spec in manifest.items():
        pure=PurePosixPath(rel);stored=PurePosixPath(spec['stored_path'])
        assert not pure.is_absolute() and '..' not in pure.parts and not stored.is_absolute() and '..' not in stored.parts
        assert pure.suffix not in forbidden and 'authorization' not in pure.name.lower() and 'submission_intent' not in pure.name.lower()
        raw=data(rel);assert secret.search(raw) is None,rel
        if pure.suffix=='.py':ast.parse(raw.decode('utf-8-sig'));python+=1
    summary=read(base/'delivery_summary.json');deferred=[];dependency_checks=0
    for folder in summary['run_catalog'].values():
        prefix='bundle/runs/'+folder+'/inputs/'
        lock=load(prefix+'execution_lock.json')
        for rel,digest in lock.get('runtime_sha256',lock.get('runtime_SHA256',{})).items():
            if rel.startswith('models/'):assert assets[rel]['sha256']==digest
            else:assert hashlib.sha256(data('bundle/'+rel)).hexdigest()==digest
            dependency_checks+=1
        for name,digest in lock.get('input_sha256',lock.get('input_SHA256',{})).items():
            rel=prefix+name
            if rel in manifest:assert hashlib.sha256(data(rel)).hexdigest()==digest
            else:
                assert external[rel]['sha256']==digest;deferred.append(rel)
    module=ast.parse(data('bundle/experiments/2b-sember-v1/sember_protocol.py').decode('utf-8'))
    functions=[node for node in module.body if isinstance(node,ast.FunctionDef) and node.name in ('parse_choice','format_ok')]
    assert len(functions)==2;scope={'re':re};exec(compile(ast.Module(body=functions,type_ignores=[]),'<pinned-mcq-parser>','exec'),scope)
    key={}
    if labels is not None:
        wanted='bundle/runs/sember_prepare_20261025_01/selection/labels_LOCAL_ONLY.json'
        assert hashlib.sha256(labels.read_bytes()).hexdigest()==external[wanted]['sha256']
        key={r['question_id']:r['correct_letter'] for r in read(labels)}
    answers=Counter();scores=defaultdict(lambda:dict(answers=0,parsed=0,format_ok=0,correct=0));labelled=0
    for rel in manifest:
        if not rel.endswith('/answers.jsonl'):continue
        rows=[json.loads(line) for line in data(rel).decode('utf-8').splitlines() if line]
        seen=set()
        for row in rows:
            qid=row.get('question_id',row.get('id'));arm=row.get('arm',row.get('condition'))
            assert qid is not None and arm is not None and (qid,arm) not in seen;seen.add((qid,arm))
            assert isinstance(row['text'],str) and 1<=len(row['generated_ids'])<=16
            answers[rel]+=1
            if 'question_id' not in row:continue  # Prior 303-question QProbe has its own parser.
            choice=scope['parse_choice'](row['text'],5);k=rel+':'+arm;s=scores[k]
            s['answers']+=1;s['parsed']+=int(choice is not None);s['format_ok']+=scope['format_ok'](row['text'],5)
            if key:
                assert qid in key;s['correct']+=choice==key[qid];labelled+=1
        if '/sember_main_prepare_' in rel and '/bounded/' in rel:assert len(rows)==5258
        if '/sember_main_prepare_' in rel and '/reference/' in rel:assert len(rows)==478
    # All-kept control and the original R1 gate remain unchanged.
    r1=load('bundle/runs/vision_midprune_prepare_20261025_01/visual_cost_analysis.json')
    assert r1['status']=='complete' and not r1['gate_passed'] and r1['question_accuracy_not_measured'] is True
    assert abs(r1['median_pair_ratio']-.8553818178784731)<1e-12
    r2=load('bundle/runs/midprune_quality_prepare_20261026_01/inputs/protocol.json')
    assert r2['max_answers']==2160 and r2['max_vision_encodings']==37014
    assert summary['status']=='checkpoint_not_R2_final_result'
    if materialize is not None:
        target=materialize.resolve();assert not target.exists();target.mkdir(parents=True)
        for rel in manifest:
            if not rel.startswith('bundle/'):continue
            p=(target/rel[7:]).resolve();assert p.is_relative_to(target);p.parent.mkdir(parents=True,exist_ok=True)
            with p.open('xb') as f:f.write(data(rel))
        (target/'EXTERNAL_INPUTS_REQUIRED.json').write_text(json.dumps(external,ensure_ascii=False,indent=2),encoding='utf-8')
    result=dict(status='passed',files=len(manifest),python_AST=python,dependency_SHA_checks=dependency_checks,
        raw_answers=sum(answers.values()),answer_streams=dict(answers),content_answers_recomputed_from_private_labels=labelled,
        official_keys_checked=bool(key),official_input_restore_deferred=len(set(deferred)),
        question_options_and_answer_keys_not_redistributed=True,R2_not_claimed_complete=True,
        real_model_loads=0,GPU_jobs=0,materialized=materialize is not None)
    if key:result['recomputed_content_format_by_stream_arm']=dict(scores)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--labels',type=Path);p.add_argument('--materialize',type=Path);p.add_argument('--output',type=Path)
    a=p.parse_args();r=verify(a.root,a.labels,a.materialize)
    if a.output:
        with a.output.open('x',encoding='utf-8') as f:json.dump(r,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:v for k,v in r.items() if k not in ('answer_streams','recomputed_content_format_by_stream_arm')},ensure_ascii=False))
