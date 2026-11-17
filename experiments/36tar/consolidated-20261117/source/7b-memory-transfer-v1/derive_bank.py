"""Create isolated grid/width adaptation; preserve the frozen LS merge arithmetic."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
def main():
    original=ROOT/'experiments/2b-resolution-cost-v1'
    replacements={
        'resolution_nodes.py':('q25_nodes.py',[
            ('((6,8),(9,12),(12,16))','((6,8),(9,12),(12,16),(10,14))'),
            ('1<=x.shape[1]<=2048','1<=x.shape[1]<=3584')]),
        'resolution_banks_base.py':('q25_banks.py',[
            ('from resolution_common import ARMS,THRESHOLD,SPECS,arms_for\n',''),
            ('from resolution_nodes import','from q25_nodes import')])}
    rows=[]
    for name,(target,edits) in replacements.items():
        before=(original/name).read_text(encoding='utf-8');after=before
        for old,new in edits:
            assert after.count(old)==1,(name,old)
            after=after.replace(old,new)
        with (HERE/target).open('x',encoding='utf-8',newline='\n') as f:
            f.write('# Adapted from experiments/2b-resolution-cost-v1/'+name+'; see bank_derivation.json.\n'+after)
        rows.append(dict(source=str((original/name).relative_to(ROOT)),target=target,
            source_sha256=hashlib.sha256((original/name).read_bytes()).hexdigest(),
            target_sha256=hashlib.sha256((HERE/target).read_bytes()).hexdigest(),edits=edits))
    (HERE/'bank_derivation.json').write_text(json.dumps(rows,indent=2)+'\n')
if __name__=='__main__':main()
