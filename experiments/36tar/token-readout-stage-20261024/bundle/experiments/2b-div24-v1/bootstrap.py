"""Build isolated P1 harness from the frozen previous experiment, never edit it."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
OLD = ROOT / 'experiments/2b-token-compression-v1'
OUT = ROOT / 'runs/2b_div24_prepare_20261023_01'


def main():
    OUT.mkdir(exist_ok=False)
    (OUT/'inputs').mkdir()
    provenance = {}
    for name in ('common.py', 'compression_observer.py', 'evaluate.py', 'freeze.py', 'run_eval.sh'):
        src = OLD/name
        text = src.read_text(encoding='utf-8')
        provenance[src.relative_to(ROOT).as_posix()] = hashlib.sha256(src.read_bytes()).hexdigest()
        text = text.replace('2b_token_compression_prepare_20261021_01', '2b_div24_prepare_20261023_01')
        text = text.replace('2b-token-compression-v1', '2b-div24-v1')
        text = text.replace("('full48', 'odd24', 'tome24')", "('full48', 'odd24', 'grid24', 'tome24', 'div24')")
        text = text.replace('compression_three', 'compression_five')
        text = text.replace('answers==909', 'answers==1515').replace('generated<=7272', 'generated<=12120')
        text = text.replace('answers_max=909', 'answers_max=1515').replace('generated_tokens_max=7272', 'generated_tokens_max=12120')
        text = text.replace('wall_seconds=5400,worker_seconds=5100', 'wall_seconds=3600,worker_seconds=3300')
        text = text.replace('started<5100', 'started<3300').replace('--time=01:30:00', '--time=01:00:00')
        text = text.replace('2b-token-pair', '2b-div24').replace('2b-token-compression-$SLURM_JOB_ID', '2b-div24-$SLURM_JOB_ID')
        (HERE/name).write_text(text, encoding='utf-8', newline='\n')
    # Context builder unchanged; no compression-specific prompt or reindexing.
    text = (OLD/'memory.py').read_text(encoding='utf-8')
    context = text[text.index('@torch.inference_mode()\ndef context'):]
    (HERE/'memory.py').write_text('''"""Finite five-arm archive; not a deployment memory bank."""
import itertools
import time
import torch
from common import ARMS
from features import raw_block, reduce
from forest import Node, raw, validate
from selection import div_indices, grid_indices


def variants(features, index, begin, end):
    block = raw_block(features, index, begin, end)
    nodes = {'full48': raw(block)}
    for arm in ARMS[1:]:
        if arm in ('odd24', 'tome24'):
            selected = reduce(block, 'odd' if arm == 'odd24' else 'tome')
            values, mass, xy = selected.features, selected.mass, selected.xy
        else:
            ids = grid_indices() if arm == 'grid24' else div_indices(block.features)
            values, mass, xy = (t[ids].clone() for t in (block.features, block.mass, block.xy))
        nodes[arm] = Node(values, mass, torch.cat((torch.full((24,1),index,dtype=torch.long),xy),1),
                          index,index,begin,end,(begin+end)/2)
        validate(nodes[arm])
    return nodes


class Archive:
    def __init__(self, pair_cap=320):
        self.pair_cap = pair_cap
        self.nodes = {a: [] for a in ARMS}
        self.reduction_seconds = {a: 0. for a in ARMS}

    @torch.inference_mode()
    def append(self, features, index, begin, end):
        if index != len(self.nodes['full48']) or index >= self.pair_cap:
            raise ValueError('finite contiguous archive required')
        if index and begin <= self.nodes['full48'][-1].end:
            raise ValueError('new real time pair required')
        started = time.monotonic()
        nodes = variants(features,index,begin,end)
        self.reduction_seconds['full48'] += time.monotonic()-started
        # Above is shared preparation timing, not a per-method performance result.
        for a in ARMS:
            self.nodes[a].append(nodes[a])
        return self.status()

    def status(self):
        return dict(pairs=len(self.nodes['full48']),
            tokens={a:sum(len(n.features) for n in ns) for a,ns in self.nodes.items()},
            tensor_bytes={a:sum(n.metadata()['tensor_bytes'] for n in ns) for a,ns in self.nodes.items()},
            diagnostic_archive_NOT_bounded_long_memory=True,
            reduction_timing='shared_five_arm_preparation_not_per_arm')

    def fingerprint(self):
        from fixed_stage2 import node_digest
        return {a:node_digest(ns) for a,ns in self.nodes.items()}

    def clear(self):
        self.nodes = {a:[] for a in ARMS}


''' + context, encoding='utf-8', newline='\n')
    source = ROOT/'runs/2b_token_compression_prepare_20261021_01'
    for name in ('groups.json','questions.json'):
        src=source/'inputs'/name
        (OUT/'inputs'/name).write_bytes(src.read_bytes())
        provenance[src.relative_to(ROOT).as_posix()] = hashlib.sha256(src.read_bytes()).hexdigest()
    for name in ('labels_LOCAL_ONLY.json','input_provenance.json'):
        (OUT/name).write_bytes((source/name).read_bytes())
    protocol=json.loads((source/'inputs/protocol.json').read_text(encoding='utf-8'))
    protocol.update(arms=['full48','odd24','grid24','tome24','div24'], answers_max=1515,
        generated_tokens_max=12120, new_candidate='DivPrune Algorithm1 greedy projection subset',
        hypothesis='Can diversity selection preserve reading quality at equal24 token count?',
        equal_odd_tome_positions=True, all_arm_positions_equal=False,
        archive='finite full plus four half versions, not bounded deployment',
        feature_archive_bytes_cap_at2048=379238400,
        selection_seed='max nearest-neighbour cosine distance excluding self',
        ties='lowest original index', zero_vectors='F.normalize eps1e-12',
        output_order='original sorted indices',
        method_reference='https://arxiv.org/html/2503.02175v2#alg1',
        input_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (OUT/'inputs').iterdir()})
    (OUT/'inputs/protocol.json').write_text(json.dumps(protocol,indent=2)+'\n',encoding='utf-8')
    (OUT/'bootstrap_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
    print('Isolated five-arm harness prepared; no GPU authorization.')


if __name__ == '__main__': main()
