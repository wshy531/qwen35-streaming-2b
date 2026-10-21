"""Second bounded metadata pass: fixed subjects/days, no media/model/GPU."""
import hashlib,json,re,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'runs/2b_memory_value_design_20261021_01'
OUT=ROOT/'runs/2b_memory_value_prepare_20261021_01'
REV='143fb319be7aa5ae210c936bf4f0f3a86092afb0'
SOURCES=(('A1_JAKE','DAY1'),('A2_ALICE','DAY2'),('A3_TASHA','DAY3'),('A4_LUCIA','DAY4'))
def save(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def stamp(path):
    value=Path(path).stem.split('_')[-1]
    assert re.fullmatch(r'\d{8}',value),path
    h,m,s=int(value[:2]),int(value[2:4]),int(value[4:6]);assert h<24 and m<60 and s<60
    return h*3600+m*60+s,value[-2:]


def main():
    OUT.mkdir(exist_ok=False);start=time.monotonic();total=0;manifest=[];sources=[];receipts=[]
    save(OUT/'selection_rule.json',dict(subject_days=SOURCES,revision=REV,files_each=480,
        rule='first480 chronological MP4 per preselected subject/day; no content/output selection',
        scope='metadata only; nominal4h needs actual durations/PTS and gap admission',
        network_cap_bytes=4*2**20,network_seconds=120,requests_max=6,no_retry=True,GPU_jobs=0))
    for subject,day in SOURCES:
        if subject=='A1_JAKE':
            data=(OLD/'day1_listing.json').read_bytes();listing=json.loads(data)
            receipts.append(dict(subject=subject,day=day,source='reuse pinned earlier listing',bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
        else:
            url=f'https://huggingface.co/api/datasets/lmms-lab/EgoLife/tree/{REV}/{subject}/{day}?limit=1000'
            with urllib.request.urlopen(url,timeout=20) as response:
                data=bytearray()
                while True:
                    block=response.read(65536)
                    if not block:break
                    data.extend(block);total+=len(block)
                    assert total<4*2**20 and time.monotonic()-start<120
                assert 'rel="next"' not in response.headers.get('Link',''),'bounded listing incomplete; stop before selecting'
                listing=json.loads(data)
            p=OUT/f'listing_{subject}_{day}.json'
            with p.open('xb') as f:f.write(data)
            receipts.append(dict(subject=subject,day=day,url=url,bytes=len(data),sha256=sha(p)))
        videos=sorted((v for v in listing if v['type']=='file' and v['path'].endswith('.mp4')),key=lambda v:stamp(v['path']))
        assert len(videos)>=480,(subject,day,len(videos))
        chosen=videos[:480];times=[stamp(v['path'])[0] for v in chosen]
        gaps=[dict(before=chosen[i-1]['path'],after=chosen[i]['path'],whole_second_difference=times[i]-times[i-1])
            for i in range(1,len(chosen)) if times[i]-times[i-1]>31]
        for i,v in enumerate(chosen):
            assert re.fullmatch(r'[0-9a-f]{64}',v['lfs']['oid']) and v['size']==v['lfs']['size']
            assert v['path'].startswith(subject+'/'+day+'/') and v['size']<100*2**20
            manifest.append(dict(subject=subject,day=day,ordinal=i,path=v['path'],bytes=v['size'],sha256=v['lfs']['oid'],
                revision=REV,url=f'https://huggingface.co/datasets/lmms-lab/EgoLife/resolve/{REV}/'+v['path']))
        sources.append(dict(subject=subject,day=day,files=480,bytes=sum(v['size'] for v in chosen),
            first_file=chosen[0]['path'],last_file=chosen[-1]['path'],nominal_clip_seconds=30,
            nominal_observed_hours=4,filename_clock_span_seconds=times[-1]-times[0]+30,
            filename_gaps_over31_seconds=gaps,actual_PTS_and_duration_verified=False,
            last_two_clock_digits='opaque pending author/actual media verification',
            independence='different wearer/day; all from same shared household, not4 independent environments'))
    assert len(manifest)==1920 and total<4*2**20
    save(OUT/'media_manifest1920.json',manifest)
    summary=dict(status='pinned_metadata_selection_NOT_actual_video_admission',sources=sources,
        files=len(manifest),bytes=sum(v['bytes'] for v in manifest),network_metadata_bytes=total,
        receipts=receipts,seconds=time.monotonic()-start,video_downloaded_bytes=0,GPU_jobs=0,
        media_manifest_sha256=sha(OUT/'media_manifest1920.json'))
    save(OUT/'source_selection_receipt.json',summary)
    print(dict(files=len(manifest),bytes=summary['bytes'],GiB=summary['bytes']/2**30,
        sources=[dict(subject=s['subject'],day=s['day'],GiB=s['bytes']/2**30,gaps=len(s['filename_gaps_over31_seconds'])) for s in sources],
        metadata_bytes=total,video_downloaded_bytes=0,GPU_jobs=0))


if __name__=='__main__':main()
