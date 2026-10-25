"""Torch checks executed only inside the approved CPU preflight."""
from .memory import Bank,raw,context,answer
from ..online_memory_2b.session import MemorySession
from .controls import PRESETS,geometry


def compare_nodes(torch,left,right):
    if len(left)!=len(right): raise AssertionError('node count differs')
    for a,b in zip(left,right):
        metadata=dict(a.metadata()); metadata.pop('grid',None)
        if metadata!=b.metadata(): raise AssertionError('baseline metadata differs')
        for key in ('features','mass','coords'):
            if not torch.equal(getattr(a,key),getattr(b,key)): raise AssertionError('baseline tensor differs: '+key)


def low_parity(torch,width=32,length=120):
    # Alternate similar runs and unrelated events to exercise within-event,
    # cross-event and depth-first selection, with deterministic generator.
    rng=torch.Generator().manual_seed(20261024)
    results={}
    for arm,oldarm in (('LS','LS_shared384_v1'),('A06','M2_A06')):
        new=Bank(arm,'low'); old=MemorySession(oldarm)
        base=None
        for i in range(length):
            if i%7==0: base=torch.randn(48,width,generator=rng)
            x=base+torch.randn(48,width,generator=rng)*.001
            new.append(raw(x,i,2.*i,2.*i+1,'low'))
            old.append_pair(old.legacy.features.raw_block(x,i,2.*i,2.*i+1),frame_ids=(2*i,2*i+1))
            compare_nodes(torch,new.nodes(),old.export())
            if (new.within!=old.bank.within_merges or new.cross!=old.bank.long_merges
                    or not torch.equal(new.previous,old.previous)):
                raise AssertionError('baseline event/merge state differs')
            if arm=='A06' and new.selection_count!=old.bank.selection_count: raise AssertionError('selector differs')
        if not new.within or not new.cross: raise AssertionError('insufficient merge coverage')
        results[arm]=(new,old)
    return results


def tiny_memory(env,model,builder,examples):
    torch=env['torch']; results=[]
    # Exact low-grid composition equivalence, including embeddings and 4-axis positions.
    for arm,(bank,old) in low_parity(torch).items():
        limits=env['Limits'](8192,384,8)
        a=context(model,builder,bank.nodes(),'Return only A.',limits=limits)
        b=env['context'](model,builder,old.export(),'Return only A.',limits=limits)
        if any(not torch.equal(x,y) for x,y in zip(a[:3],b[:3])):
            raise AssertionError('192x256 readout composition changed')
        old.close()
    for resolution in PRESETS:
        # Exercise the streaming vision implementation on real RGB at every
        # new grid, rather than testing the bank only with random features.
        from .native import inputs
        packets,metadata=examples[resolution]
        native=inputs(env,model,builder,packets,'Return only A.',metadata)
        features=torch.cat(model.model.get_video_features(native.pixel_values_videos,
            native.video_grid_thw,return_dict=True).pooler_output,0)
        if features.shape!=(geometry(resolution)['pair_tokens'],32) or not torch.isfinite(features).all():
            raise AssertionError('tiny streaming vision grid mismatch')
        del native
        for arm in ('LS','A06'):
            bank=Bank(arm,resolution)
            for i in range(20):
                bank.append(raw(features+torch.randn_like(features)*.001,i,2.*i,2.*i+1,resolution))
            state=bank.digest()
            result=answer(env,model,builder,bank,'Return only A.')
            if bank.digest()!=state: raise AssertionError('tiny question changed bank')
            results.append(dict(arm=arm,resolution=resolution,generated_ids=result['generated_ids']))
            bank.reset()
            if bank.digest()!=Bank(arm,resolution).digest(): raise AssertionError('tiny reset')
    return results


def tiny_native_parity(env,model,native):
    """Compare complete tiny native prefills, not a substitute for GPU replay."""
    from .native import query_tiling
    torch=env['torch']; rope=model.model.rope_deltas
    try:
        expected=model(**native,use_cache=False,logits_to_keep=1).logits.clone()
        model.model.rope_deltas=rope
        with query_tiling(model,tile=8):
            actual=model(**native,use_cache=False,logits_to_keep=1).logits
        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
        if not torch.equal(actual.argmax(-1),expected.argmax(-1)):
            raise AssertionError('tiled tiny native argmax changed')
    finally:
        model.model.rope_deltas=rope
