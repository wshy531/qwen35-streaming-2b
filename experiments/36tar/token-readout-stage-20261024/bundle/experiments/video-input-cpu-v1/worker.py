"""One guarded local CPU decode case; no neural inference or GPU allocation."""
import argparse,os,sys,time,platform,importlib.util
from pathlib import Path
import psutil
from common import ROOT,OUT,read,save,sha
from source import BoundedAVSource
from consume import consume


def main(index):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1' and 'torch' not in sys.modules
    run=OUT/'inputs';lock=read(run/'execution_lock.json')
    for name,digest in lock['runtime_SHA256'].items():assert sha(ROOT/name)==digest,name
    for name,digest in lock['input_SHA256'].items():assert sha(run/name)==digest,name
    case=read(run/'cases.json')[index];assert case['index']==index
    group=next(g for g in read(run/'groups.json') if g['id']==case['group'])
    questions=[q for q in read(run/'questions.json') if q['group']==group['id']]
    path=(ROOT/group['local_media']).resolve();assert path.is_relative_to(ROOT.resolve()) and path.is_file()
    process=psutil.Process();started=time.monotonic();max_rss=process.memory_info().rss;max_threads=process.num_threads()
    def guard():
        nonlocal max_rss,max_threads
        info=process.memory_info();max_rss=max(max_rss,info.rss);max_threads=max(max_threads,process.num_threads())
        assert max_rss<1536*2**20 and time.monotonic()-started<230
    source=BoundedAVSource(path,group,config=case['config'],guard=guard)
    try:
        queries=consume(source,group,questions,read(run/'snapshots34.json'));guard()
        assert not source.buffer and source.completed
        elapsed=time.monotonic()-started;info=process.memory_info()
        peak=getattr(info,'peak_wset',max_rss)
        assert peak<1536*2**20 and 'torch' not in sys.modules
        report=dict(status='passed',case=case,source_SHA256=group['sha256'],source=source.status(),
            input_wall_seconds=elapsed,query_checks=queries,query_count=len(queries),
            sampled_pixels_and_PTS_all_verified=True,peak_RSS_bytes=peak,max_observed_threads=max_threads,
            all_buffers_released=True,pretrained_loads=0,GPU_jobs=0,
            dependency=dict(Python=sys.version,platform=platform.platform(),decord_available=importlib.util.find_spec('decord') is not None))
        save(OUT/('case-'+str(index)+'.json'),report)
        print(dict(case=index,config=case['config'],seconds=round(elapsed,3),RSS_MiB=round(peak/2**20,1)),flush=True)
    finally:source.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--case',type=int,required=True)
    main(parser.parse_args().case)
