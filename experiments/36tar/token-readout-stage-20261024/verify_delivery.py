"""Standard-library delivery verification; no GPU, network or model loading."""
import argparse,ast,gzip,hashlib,json,re
from pathlib import Path,PurePosixPath

def main(base,materialize=None,model_dir=None):
    base=base.resolve();manifest=json.loads((base/'manifest.json').read_text(encoding='utf-8'))
    assets=json.loads((base/'external_assets.json').read_text(encoding='utf-8'))
    if model_dir is not None:
        for rel,spec in assets.items():
            p=model_dir/PurePosixPath(rel).name
            assert hashlib.sha256(p.read_bytes()).hexdigest()==spec['sha256'],rel
    def data(relative):
        spec=manifest[relative];raw=(base/spec['stored_path']).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==spec['sha256'],relative
        decoded=gzip.decompress(raw) if spec['gzip'] else raw
        assert len(decoded)==spec['original_bytes'] and hashlib.sha256(decoded).hexdigest()==spec['original_sha256'],relative
        return decoded
    def read(relative):return json.loads(data(relative).decode('utf-8'))
    py=0;total=0
    forbidden={'.mp4','.webm','.mkv','.jpg','.png','.safetensors','.pt','.pth','.npz','.npy','.key','.pem'}
    for relative,spec in manifest.items():
        rel=PurePosixPath(relative);stored=PurePosixPath(spec['stored_path'])
        assert not rel.is_absolute() and '..' not in rel.parts
        assert not stored.is_absolute() and '..' not in stored.parts
        assert rel.suffix not in forbidden and rel.name!='GPU_authorization.json'
        payload=data(relative);total+=len(payload)
        if rel.suffix=='.py':ast.parse(payload.decode('utf-8-sig'));py+=1
    catalog=json.loads((base/'delivery_summary.json').read_text(encoding='utf-8'))['run_catalog']
    for folder in catalog.values():
        lock=read('bundle/runs/'+folder+'/inputs/execution_lock.json')
        for rel,digest in lock.get('runtime_sha256',lock.get('runtime_SHA256',{})).items():
            if rel.startswith('models/'):
                assert assets[rel]['sha256']==digest,rel
                continue
            assert hashlib.sha256(data('bundle/'+rel)).hexdigest()==digest,rel
        if folder in ('2b_auto4_cost_prepare_20261024_01','2b_readout_compression_prepare_20261024_01'):
            for rel,digest in lock['input_sha256'].items():
                assert hashlib.sha256(data('bundle/runs/'+folder+'/inputs/'+rel)).hexdigest()==digest,rel
    e1=read('bundle/runs/2b_auto4_cost_prepare_20261024_01/cost_and_replay_results.json')
    e2=read('bundle/runs/2b_readout_compression_prepare_20261024_01/integrity_and_decision.json')
    assert e1['replay_mismatches']==[] and e2['legacy_replay']==909 and e2['answers']==1818
    # Re-run only the pinned parser definitions, never experiment/module startup code.
    parser=ast.parse(data('bundle/experiments/2b-readout-compression-v1/analyze.py').decode('utf-8'))
    definitions=[n for n in parser.body if isinstance(n,ast.FunctionDef) and n.name in ('norm','parse') or
        isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('STRICT','LEADING','EXPLICIT') for t in n.targets)]
    scope={'re':re};exec(compile(ast.Module(body=definitions,type_ignores=[]),'<pinned-parser>','exec'),scope)
    verified_scores=0
    for folder,names in (('2b_auto4_cost_prepare_20261024_01',['answer_results_repeat0.json','answer_results_repeat1.json']),
                         ('2b_readout_compression_prepare_20261024_01',['answer_results.json'])):
        prefix='bundle/runs/'+folder+'/'
        qs={q['id']:q for q in read(prefix+'inputs/questions.json')}
        labels={q['id']:q for q in read(prefix+'labels_LOCAL_ONLY.json')}
        for name in names:
            result=read(prefix+name)
            for row in result['details']:
                parsed=scope['parse'](row['text'],qs[row['id']]['options'],status='ok')
                assert row['answer_correct']==(parsed['choice']==labels[row['id']]['answer_index'])
                assert row['format_compliant']==parsed['format_compliant'];verified_scores+=1
            for subset,table in result['tables'].items():
                for arm,score in table['scores'].items():
                    assert 0<=score['correct']<=score['denominator']
    if materialize is not None:
        target=materialize.resolve();assert not target.exists(),'materialization requires a new directory'
        target.mkdir(parents=True)
        for relative in manifest:
            if not relative.startswith('bundle/'):continue
            dest=target/relative[7:];assert dest.resolve().is_relative_to(target)
            dest.parent.mkdir(parents=True,exist_ok=True)
            with dest.open('xb') as f:f.write(data(relative))
    print(json.dumps(dict(status='passed',files=len(manifest),python_AST=py,uncompressed_bytes=total,
        content_answers_recomputed=verified_scores,pretrained_loads=0,GPU_jobs=0,
        no_media_weights_secrets=True,external_model_metadata_verified=model_dir is not None,
        materialized=materialize is not None),ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--materialize',type=Path);p.add_argument('--model-dir',type=Path)
    a=p.parse_args();main(a.root,a.materialize,a.model_dir)
