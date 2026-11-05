"""Manual bounded completion of whole missing sources only, same frozen worker.

Not used unless the original allocation has ended. The new source list and
budget need their own execution lock/receipt within the user's work window.
No in-source resume, changed algorithm, auto resubmission, or new questions.
"""
import argparse,os,sys,time
from pathlib import Path
def main(prep,index,out):
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('approved Slurm allocation required')
    import common
    assert prep.resolve().is_relative_to((common.ROOT/'runs').resolve())
    common.PREP=prep
    import worker
    worker.PREP=prep
    common.verify()
    a=common.read(prep/'GPU_authorization.json')
    assert a['approved'] and a['source_completion_only'] and a['minutes']==10
    assert out.resolve().is_relative_to((common.ROOT/'runs').resolve())
    out.parent.mkdir(exist_ok=True)
    worker.main(index,out,time.time()+10*60-60)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prep',type=Path,required=True);p.add_argument('--index',type=int,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.prep,a.index,a.out)
