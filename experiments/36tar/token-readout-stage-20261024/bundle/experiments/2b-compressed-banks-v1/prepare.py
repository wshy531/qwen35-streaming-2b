"""Build bounded P3 adapters; frozen previous implementations remain untouched."""
from pathlib import Path
import ast,json,hashlib
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).parent
OUT=ROOT/'runs/2b_compressed_banks_prepare_20261023_01'
OUT.mkdir(exist_ok=False)
old=ROOT/'experiments/2b-ls-depth-v1'
common=(old/'common.py').read_text(encoding='utf-8').replace('2b_ls_depth_prepare_20261021_01','2b_compressed_banks_prepare_20261023_01')
common=common.replace("ARMS = ('M0', 'reservoir4', 'M2', 'A06', 'LS', 'LS_depth')", "ARMS = tuple(p+'_'+c for p in ('M0','M2','A06','LS') for c in ('full48','change24'))+('M0_odd24',)")
common+="\nsys.path.append(str(ROOT/'experiments/2b-ls-depth-v1'))\n"
(HERE/'common.py').write_text(common,encoding='utf-8',newline='\n')
(HERE/'selection.py').write_bytes((ROOT/'experiments/2b-temporal24-v1/selection.py').read_bytes())
def method(file,cls,name):
    text=file.read_text(encoding='utf-8');tree=ast.parse(text)
    c=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name==cls)
    m=next(x for x in c.body if isinstance(x,ast.FunctionDef) and x.name==name)
    return '\n'.join(text.splitlines()[m.decorator_list[0].lineno-1:m.end_lineno])+'\n'
forest=method(ROOT/'experiments/2b-natural-eval-v1/memory384.py','Forest384','append')
forest=forest.replace('if len(self.recent)==4:', 'if sum(len(n.features) for n,_ in self.recent)>=192:')
forest=forest.replace("i=min(range(len(costs)),key=lambda i:costs[i]['total'])",'i=self.select(costs)')
shared=method(old/'shared.py','SharedHistory384','append')
shared=shared.replace('if len(self.recent) == 4:', 'if sum(len(e.node.features) for e in self.recent) >= 192:')
header='''"""Token-budget adapters; original merge/penalties unchanged, no global patch."""
from types import SimpleNamespace
import torch
from memory384 import Forest384
from shared import SharedHistory384,EventNode
from legacy_sources import load
from forest import Node,validate
from features import validate as validate_block

def pair_node(block):
    validate_block(block)
    if block.first!=block.last or len(block.features) not in (24,48) or not bool(torch.all(block.mass==1)):
        raise ValueError('one selected or full original pair required')
    n=Node(block.features.clone(),block.mass.clone(),
        torch.cat((torch.full((len(block.features),1),block.first,dtype=torch.long),block.xy.clone()),1),
        block.first,block.last,block.begin,block.end,block.center)
    validate(n)
    return n

def adapter():
    original=load().forest
    return SimpleNamespace(raw=pair_node,merge=original.merge,penalties=original.penalties)

class BoundedForest(Forest384):
    def __init__(self,depth_first=False):
        super().__init__()
        self.legacy=adapter();self.depth_first=depth_first
    def select(self,costs):
        if not self.depth_first:return min(range(len(costs)),key=lambda i:costs[i]['total'])
        return min(range(len(costs)),key=lambda i:(1+max(self.roots[i].depth,self.roots[i+1].depth),costs[i]['total'],i))
'''
header+=forest
header+='''
class BoundedShared(SharedHistory384):
    def __init__(self):
        super().__init__()
        self.legacy=adapter()
'''+shared
(HERE/'policy_adapters.py').write_text(header,encoding='utf-8',newline='\n')
reader=(ROOT/'experiments/2b-natural-eval-v1/readout.py').read_text(encoding='utf-8')
reader=reader.replace("if node.metadata()['exact_pair'] else",'if node.first == node.last else')
(HERE/'bank_readout.py').write_text(reader,encoding='utf-8',newline='\n')
provenance={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
    (ROOT/'experiments/2b-natural-eval-v1/memory384.py',old/'shared.py',old/'common.py',
     ROOT/'experiments/2b-natural-eval-v1/readout.py',ROOT/'experiments/2b-temporal24-v1/selection.py')}
(OUT/'adapter_sources.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
print('P3 independent adapters generated; no GPU or model loading')
