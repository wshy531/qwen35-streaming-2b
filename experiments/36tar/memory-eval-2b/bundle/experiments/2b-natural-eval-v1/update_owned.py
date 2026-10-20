"""Archive then repair only known files in this run's private adapter folder."""
import base64,hashlib,json,sys
from pathlib import Path
from cluster_io import command,LOCAL,HERE
def main():
    receipt=json.loads((LOCAL/'transfer_receipt.json').read_text())
    old={x['path']:x['sha256'] for x in receipt['files']}
    payload=[]
    for rel in sys.argv[1:]:
        if rel=='source_lock.json':p=LOCAL/rel;remote='runs/2b-natural-eval-inputs-20261020-01/'+rel
        else:p=HERE/rel;remote='experiments/2b-natural-eval-v1/'+rel
        payload.append(dict(path=remote,old_sha256=old[remote],content=base64.b64encode(p.read_bytes()).decode()))
    code='payload='+repr(payload)+'''
import pathlib,hashlib,base64,json
p=pathlib.Path.cwd();archive=p/'runs/2b-natural-eval-inputs-20261020-01/engineering_fixes';archive.mkdir(exist_ok=True)
saved=[]
for item in payload:
 dst=p/item['path']
 if not dst.resolve().is_relative_to(p.resolve()):raise ValueError('outside project')
 previous=dst.read_bytes();h=hashlib.sha256(previous).hexdigest()
 if h!=item['old_sha256']:raise ValueError('existing owned file drift')
 target=archive/(dst.name+'.'+h[:12])
 if not target.exists():target.write_bytes(previous)
 value=base64.b64decode(item['content']);dst.write_bytes(value)
 saved.append(dict(path=item['path'],before=h,after=hashlib.sha256(value).hexdigest()))
print(json.dumps(saved))
'''
    s=command(code);(LOCAL/'engineering_update_01.json').write_text(s,encoding='utf-8');print(s)
if __name__=='__main__':main()
