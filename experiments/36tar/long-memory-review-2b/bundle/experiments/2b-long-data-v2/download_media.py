"""Authorized fixed-manifest, two-worker public download; no GPU or auto retry."""
import argparse,concurrent.futures,hashlib,json,os,threading,time,urllib.request
from pathlib import Path,PurePosixPath
ROOT=Path(__file__).resolve().parents[2];RUN=ROOT/'runs/2b_memory_value_prepare_20261021_01'
DEST=ROOT/'datasets/2b-memory-eval/egolife-memory4-20261021'
CAP=30*2**30;SECONDS=10800;WORKERS=2
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2)


def main(approved):
    manifest=RUN/'media_manifest1920.json';rows=read(manifest);h=sha(manifest)
    assert approved==h and len(rows)==1920 and sum(r['bytes'] for r in rows)<CAP
    auth=read(RUN/'download_authorization.json')
    assert auth['approved'] and auth['manifest_sha256']==h and auth['video_transfer_cap_bytes']==CAP
    for row in rows:
        rel=PurePosixPath(row['path']);assert not rel.is_absolute() and '..' not in rel.parts
        assert row['url'].startswith('https://huggingface.co/datasets/lmms-lab/EgoLife/resolve/143fb319be7aa5ae210c936bf4f0f3a86092afb0/')
    save(RUN/'download_attempt.json',dict(manifest_sha256=h,automatic_retry=False,workers=WORKERS,seconds_cap=SECONDS,bytes_cap=CAP))
    DEST.mkdir(parents=True,exist_ok=True);start=time.monotonic();lock=threading.Lock();stop=threading.Event();used=done=reused=0
    log=(RUN/'download_events.jsonl').open('x',encoding='utf-8')
    def fetch(row):
        nonlocal used,done,reused
        rel=PurePosixPath(row['path']);p=DEST.joinpath(*rel.parts)
        assert p.resolve().is_relative_to(DEST.resolve())
        if stop.is_set():return
        if p.exists():
            assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256']
            with lock:reused+=1;done+=1
            return
        p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix('.mp4.partial')
        assert not temp.exists(),'partial exists: manual review before resume'
        digest=hashlib.sha256();n=0
        try:
            with urllib.request.urlopen(row['url'],timeout=45) as response,temp.open('xb') as f:
                assert response.status==200
                length=response.headers.get('Content-Length')
                if length is not None:assert int(length)==row['bytes']
                while True:
                    if stop.is_set():raise RuntimeError('another transfer failed; stopping')
                    assert time.monotonic()-start<SECONDS,'download wall budget exceeded'
                    block=response.read(1048576)
                    if not block:break
                    n+=len(block);assert n<=row['bytes']
                    with lock:
                        used+=len(block);assert used<=CAP
                    digest.update(block);f.write(block)
            assert n==row['bytes'] and digest.hexdigest()==row['sha256']
            assert not p.exists();temp.rename(p)
            with lock:
                done+=1;log.write(json.dumps(dict(path=row['path'],bytes=n,sha256=row['sha256'],status='verified'))+'\n');log.flush()
                if done%20==0:print(dict(completed=done,downloaded_bytes=used,seconds=round(time.monotonic()-start,1)),flush=True)
        except BaseException:
            stop.set();raise
    failure=None
    try:
        # First two files per source are transfer/format sentinels, without content-based selection.
        sentinels=[r for r in rows if r['ordinal']<2]
        for r in sentinels:fetch(r)
        import av
        probes=[]
        for row in sentinels:
            path=DEST.joinpath(*PurePosixPath(row['path']).parts)
            with av.open(str(path)) as container:
                stream=container.streams.video[0];stream.codec_context.thread_count=1
                first=next(container.decode(stream));assert first.time is not None
                duration=float(stream.duration*stream.time_base) if stream.duration is not None else container.duration/av.time_base
                assert 0<duration<=45 and stream.width>0 and stream.height>0
                probes.append(dict(path=row['path'],duration_s=duration,average_rate=str(stream.average_rate),
                    width=stream.width,height=stream.height,first_pts_s=float(first.time),sha256=row['sha256']))
        save(RUN/'sentinel_media_probes.json',dict(status='8_files_hash_and_decode_header_passed',files=probes,
            full_PTS_or_content_review_done=False))
        rest=[r for r in rows if r['ordinal']>=2]
        with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
            pending=set();it=iter(rest)
            for _ in range(WORKERS):pending.add(pool.submit(fetch,next(it)))
            while pending:
                completed,pending=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
                for future in completed:
                    future.result()
                    row=next(it,None)
                    if row is not None:pending.add(pool.submit(fetch,row))
    except BaseException as e:
        stop.set();failure=type(e).__name__
    finally:log.close()
    save(RUN/'download_receipt.json',dict(status='failed' if failure else 'complete',failure_type=failure,
        verified_files=done,reused_files=reused,network_bytes=used,seconds=time.monotonic()-start,
        destination=str(DEST),manifest_sha256=h,automatic_retry=False,GPU_jobs=0))
    assert failure is None and done==1920,'transfer stopped; inspect receipt, no automatic retry'
    print(dict(status='complete',files=done,network_bytes=used,seconds=time.monotonic()-start))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--approved-manifest-sha256',required=True);main(p.parse_args().approved_manifest_sha256)
