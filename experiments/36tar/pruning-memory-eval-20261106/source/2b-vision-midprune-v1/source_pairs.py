"""Keep only twelve frozen real pairs; never retain a complete sampled-video archive."""
from pathlib import Path
import os
from common import ROOT,OUT,read
from sember_source import CausalSource,PairDriver

REV='897886cc967b2d0e6e45f1200605f5d3b0ffbb23'


def media_root():
    local=ROOT/'datasets/S-EMBER'/REV
    partner=Path('/home/u2025010462/share/czy/wsy/data/S-EMBER')/REV
    root=Path(os.environ.get('MIDPRUNE_MEDIA_ROOT',str(local if os.name=='nt' else partner)))
    if root.resolve() not in (local.resolve(),partner.resolve()):raise ValueError('only known own or user-provided read-only snapshot allowed')
    return root


def load_pairs():
    videos=read(OUT/'inputs/videos.json');specs=read(OUT/'inputs/pairs.json');result=[]
    for video in videos:
        wanted={s['pair_index']:s for s in specs if s['video_id']==video['video_id']}
        path=(media_root()/video['relative_path']).resolve()
        assert path.is_relative_to(media_root().resolve()) and path.stat().st_size==video['bytes']
        index=0
        def collect(first,second,*,padding=False):
            nonlocal index
            if index in wanted:
                spec=wanted[index]
                assert not padding and abs(first.timestamp-spec['first_sample_time'])<1e-9
                assert abs(second.timestamp-spec['second_sample_time'])<1e-9
                result.append(dict(spec=spec,first=first,second=second))
            index+=1
        driver=PairDriver(CausalSource(path),collect)
        try:driver.advance(max(s['query_time'] for s in wanted.values()))
        finally:driver.close()
    assert len(result)==12
    used=sum(item[k].rgb.nbytes for item in result for k in ('first','second'))
    assert used<8*2**20
    return result,used
