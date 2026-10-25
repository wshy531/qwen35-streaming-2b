"""Materialize an isolated adapter; execution never rewrites frozen modules."""
from pathlib import Path
import hashlib,json,difflib
ROOT=Path(__file__).resolve().parents[2]
src=ROOT/'experiments/2b-sember-v1';dst=ROOT/'experiments/2b-resolution-cost-v1'
rules={
 'sember_nodes.py':('resolution_nodes.py',[("((6,8),(9,12))","((6,8),(9,12),(12,16))")]),
 'sember_runtime.py':('resolution_runtime.py',[("from sember_nodes import validate","from resolution_nodes import validate"),("((192,256),(288,384))","((192,256),(288,384),(384,512))")]),
 'sember_source.py':('resolution_source.py',[("((192,256),(288,384))","((192,256),(288,384),(384,512))")]),
 'sember_session.py':('resolution_session.py',[
  ('from sember_banks import Banks','from resolution_banks import Banks'),
  ('from sember_nodes import digest','from resolution_nodes import digest'),
  ('from sember_runtime import context','from resolution_runtime import context'),
  ('from sember_common import ARMS','from resolution_common import ARMS'),
  ("*,mode='bounded'):","*,mode='bounded',arms=None):"),
  ("if mode not in ('bounded','reference')", "if mode!='bounded'"),
  ('self.bank=Banks(self.grid)','self.bank=Banks(self.grid,arms=arms)'),
  ("arm not in ARMS","arm not in self.bank.banks")]),
}
receipts={}
for name,(target,replacements) in rules.items():
    old=(src/name).read_text(encoding='utf-8');new=old
    for a,b in replacements:
        assert new.count(a)==1,(name,a,new.count(a));new=new.replace(a,b)
    new='# Isolated derivative; source and exact changes recorded in adapter_derivation.json.\n'+new
    with (dst/target).open('x',encoding='utf-8',newline='\n') as f:f.write(new)
    receipts[target]=dict(source=str((src/name).relative_to(ROOT)),source_sha=hashlib.sha256((src/name).read_bytes()).hexdigest(),
        diff=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile=name,tofile=target)))
old=(src/'sember_banks.py').read_text(encoding='utf-8')
prefix=old.split('\nclass Banks:')[0].replace('from sember_nodes import','from resolution_nodes import')
prefix=prefix.replace('from sember_common import ARMS,THRESHOLD','from resolution_common import ARMS,THRESHOLD,SPECS,arms_for')
with (dst/'resolution_banks_base.py').open('x',encoding='utf-8',newline='\n') as f:f.write(prefix+'\n')
receipts['resolution_banks_base.py']=dict(source='experiments/2b-sember-v1/sember_banks.py',
    source_sha=hashlib.sha256((src/'sember_banks.py').read_bytes()).hexdigest(),
    change='Only Bank/Entry/select/distance definitions retained; imports target isolated nodes/common. Bank algorithm unchanged.')
with (Path(__file__).parent/'adapter_derivation.json').open('x',encoding='utf-8') as f:json.dump(receipts,f,indent=2)
print(dict(status='isolated_sources_created',files=len(receipts),old_files_modified=0))
