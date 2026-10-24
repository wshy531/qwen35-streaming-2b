"""CPU-owned storage and serialized control metadata; no weights or GPU."""
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).parent
sys.path.insert(0,str(ROOT/'experiments/2b-ls-depth-v1'))
from common import ARMS,save
from banks import Banks
from preflight import owned_tensor_bytes
import torch


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
    torch.set_num_threads(1);assert not torch.cuda.is_available()
    generator=torch.Generator().manual_seed(20261021)
    banks=Banks()
    for i in range(24):banks.append(torch.randn(48,2048,generator=generator),i,float(2*i),float(2*i+1))
    state=banks.status();ledger={}
    for arm in ARMS:
        bank=banks.banks[arm]
        controls={k:v for k,v in vars(bank).items() if isinstance(v,(int,float,bool,str,type(None)))}
        if arm in ('LS','LS_depth'):controls.update(bank.controls())
        elif arm in ('M2','A06'):controls['recent_event_starts']=[flag for _,flag in bank.recent]
        elif arm=='reservoir4':controls['RNG_state']=bank.rng.getstate()
        elif arm=='M0':
            last=bank.bank.last
            controls['fifo_capacity']=bank.bank.capacity
            controls['fifo_last']=None if last is None else [last.index,last.start_s,last.end_s]
        metadata=len(json.dumps(dict(status=state['banks'][arm],controls=controls)).encode())
        assert metadata<32768
        payload=state['banks'][arm]['resident_tensor_bytes']
        assert owned_tensor_bytes(bank)==payload
        descriptor=8192 if arm in ('M2','A06','LS','LS_depth') else 0
        ledger[arm]=dict(feature_mass_coordinate_storage_bytes=payload,
            standalone_previous_descriptor_bytes=descriptor,serialized_metadata_bytes=metadata,
            sum_of_these_payload_bytes=payload+descriptor+metadata)
    total=owned_tensor_bytes(banks)
    assert total==sum(v['feature_mass_coordinate_storage_bytes'] for v in ledger.values())+8192
    banks.clear();assert owned_tensor_bytes(banks)==0
    save(HERE/'storage_accounting.json',dict(status='passed',width=2048,synthetic_pairs=24,
        six_bank_persistent_tensor_bytes=total,per_arm=ledger,cleared_storage_bytes=0,
        metadata_is_serialized_payload_not_Python_heap_measurement=True,
        source_readout_export_copies_and_append_scratch_are_transient_not_in_this_ledger=True,
        descriptor_shared_once_in_six_arm_process_but_charged_to_each_standalone_event_method=True,
        pretrained_loads=0,GPU_submissions=0))
    print(dict(status='passed',tensor_bytes=total,metadata_bytes={a:v['serialized_metadata_bytes'] for a,v in ledger.items()},GPU_submissions=0))


if __name__=='__main__':main()
