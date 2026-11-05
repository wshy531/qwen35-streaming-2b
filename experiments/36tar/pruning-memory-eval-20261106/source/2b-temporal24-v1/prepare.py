"""Build isolated C2 harness from the fixed P1 implementation."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).parent
OLD=ROOT/'experiments/2b-div24-v1'
OUT=ROOT/'runs/2b_temporal24_prepare_20261023_01'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
OUT.mkdir(exist_ok=False);(OUT/'inputs').mkdir()
for name in ('common.py','compression_observer.py','evaluate.py','run_eval.sh','analyze.py'):
    text=(OLD/name).read_text(encoding='utf-8')
    text=text.replace('2b_div24_prepare_20261023_01','2b_temporal24_prepare_20261023_01')
    text=text.replace('2b-div24-v1','2b-temporal24-v1').replace('2b-div24-$SLURM_JOB_ID','2b-temporal24-$SLURM_JOB_ID')
    text=text.replace('2b-div24','2b-temporal24').replace('compression_five','compression_temporal')
    text=text.replace("('full48', 'odd24', 'grid24', 'tome24', 'div24')","('odd24', 'change24', 'mix24')")
    text=text.replace('answers==1515','answers==909').replace('generated<=12120','generated<=7272')
    text=text.replace('expected_answers=1515','expected_answers=909').replace('expected=1515','expected=909')
    text=text.replace('three isolated fresh QA paths','three isolated fresh QA paths')
    text=text.replace("before = self.core.status()","before = self.core.status()\n        selector_before = self.archive.selector_digest()")
    text=text.replace('assert self.core.status() == before and self.archive.fingerprint() == fingerprints',
        'assert self.core.status() == before and self.archive.fingerprint() == fingerprints\n        assert self.archive.selector_digest() == selector_before')
    if name=='analyze.py':text=text.replace("'one seed reservoir',",'')
    (HERE/name).write_text(text,encoding='utf-8',newline='\n')
source=ROOT/'runs/2b_div24_prepare_20261023_01'
for name in ('groups.json','questions.json','snapshots34.json'):
    (OUT/'inputs'/name).write_bytes((source/'inputs'/name).read_bytes())
(OUT/'labels_LOCAL_ONLY.json').write_bytes((source/'labels_LOCAL_ONLY.json').read_bytes())
p=json.loads((source/'inputs/protocol.json').read_text())
p.update(arms=['odd24','change24','mix24'],answers_max=909,generated_tokens_max=7272,
    new_candidate='causal fixed12 spatial anchors plus12 largest temporal change',
    hypothesis='Does spatial protection help compared with change-only at equal24?',
    all_arm_positions_equal=False,equal_odd_tome_positions=False,
    selection_seed='no learned seed; fixed checkerboard half of odd24 lattice',
    spatial12_indices=[i for i in range(48) if i%2==1 and (i//8+(i%8)//2)%2==0],
    ties='lowest original index',cold_start='all three arms identical odd24',
    previous_pair_feature_bytes=393216,previous_pair_is_question_evidence=False,
    feature_archive_bytes_cap_at2048=316032000,
    archive='finite full48 plus three half representations; full48 only observation/layout reference, not fourth QA',
    method_reference='local Token压缩.md sections7.3-7.5 DTD; fixed quotas are project adaptation',
    input_sha256={f.name:sha(f) for f in (OUT/'inputs').iterdir()})
(OUT/'inputs/protocol.json').write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
memory=(OLD/'memory.py').read_text(encoding='utf-8')
context=memory[memory.index('@torch.inference_mode()\ndef context'):]
(HERE/'memory.py').write_text('''"""Finite archives plus one explicitly bounded causal selector state."""
import itertools,time,hashlib
import torch
from common import ARMS
from features import raw_block
from forest import Node,raw,validate
from selection import select

class Archive:
    def __init__(self,pair_cap=320):
        self.pair_cap=pair_cap
        self.nodes={a:[] for a in ('full48',)+ARMS}
        self.previous=None
        self.reduction_seconds={a:0. for a in ('full48',)+ARMS}

    def selector_digest(self):
        return None if self.previous is None else hashlib.sha256(self.previous.numpy().tobytes()).hexdigest()

    @torch.inference_mode()
    def append(self,features,index,begin,end):
        if index!=len(self.nodes['full48']) or index>=self.pair_cap:
            raise ValueError('finite contiguous archive required')
        if index and begin<=self.nodes['full48'][-1].end:
            raise ValueError('new real pair required')
        block=raw_block(features,index,begin,end)
        started=time.monotonic()
        indices,_=select(block.features,self.previous)
        nodes={'full48':raw(block)}
        for arm,ids in indices.items():
            nodes[arm]=Node(block.features[ids].clone(),block.mass[ids].clone(),
                torch.cat((torch.full((24,1),index,dtype=torch.long),block.xy[ids]),1),
                index,index,begin,end,(begin+end)/2)
            validate(nodes[arm])
        elapsed=time.monotonic()-started
        for arm,node in nodes.items():self.nodes[arm].append(node)
        self.previous=block.features.clone()
        self.reduction_seconds['full48']+=elapsed
        return self.status()

    def status(self):
        return dict(pairs=len(self.nodes['full48']),
            tokens={a:sum(len(n.features) for n in ns) for a,ns in self.nodes.items()},
            tensor_bytes={a:sum(n.metadata()['tensor_bytes'] for n in ns) for a,ns in self.nodes.items()},
            selector_bytes=0 if self.previous is None else self.previous.numel()*self.previous.element_size(),
            diagnostic_archive_NOT_bounded_long_memory=True,
            reduction_timing='shared_three_arm_preparation_not_per_arm')

    def fingerprint(self):
        from fixed_stage2 import node_digest
        return {a:node_digest(ns) for a,ns in self.nodes.items()}

    def clear(self):
        self.nodes={a:[] for a in ('full48',)+ARMS}
        self.previous=None

''' + context,encoding='utf-8',newline='\n')
print('Prepared C2 three-arm harness without GPU authorization')
