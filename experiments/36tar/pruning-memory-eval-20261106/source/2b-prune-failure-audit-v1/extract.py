"""CPU-only exact old sampler replay. Images are evidence, not token-deletion maps."""
import hashlib,json,sys
from collections import deque
from pathlib import Path
from PIL import Image,ImageDraw
R=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(R/'experiments/streamingbench-router-v1'))
from v0_source import V0Pairs
OUT=R/'runs/prune_failure_inputs_20261103_01'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    assert sys.platform!='win32','read existing cluster media; do not download videos'
    data=R/'datasets/StreamingBench-eval200-v1/videos'
    cases=json.loads((OUT/'cases.json').read_text(encoding='utf-8'))
    media={m['name']:m for m in json.loads((R/'runs/content_confirm_prepare_20261101_01/media.json').read_text())}
    assert len(cases)==6 and not (OUT/'manifest.json').exists()
    manifest=[]
    for number,q in enumerate(cases,1):
        path=data/q['media_name'];assert path.resolve().parent==data.resolve() and sha(path)==media[path.name]['sha256']
        retained=deque(maxlen=8);prefix=[];pairs=0
        def receive(a,b,padding):
            nonlocal pairs
            item=(pairs,a,b,padding);retained.append(item)
            if q['task_type']=='Counting':
                assert len(prefix)<32;prefix.append(item)
            pairs+=1
        driver=V0Pairs(path,receive,config='auto4')
        try:state=driver.advance(q['query_time_s'])
        finally:driver.close()
        assert retained and all(b.timestamp<=q['query_time_s']+1e-9 for _,a,b,_ in retained)
        groups={'actual_last8_pairs':list(retained)}
        if prefix:groups['diagnostic_complete_sampled_prefix']=prefix
        folder=OUT/f'case-{number:02d}';folder.mkdir()
        entry=dict(question=q,state=state,pairs_received=pairs,media_sha256=sha(path),groups={},images=[])
        for name,items in groups.items():
            frames=[];records=[]
            for index,a,b,padding in items:
                for offset,packet in enumerate((a,b)):
                    frames.append(packet)
                    records.append(dict(pair_index=index,within_pair=offset,frame_id=packet.frame_id,pts_s=packet.timestamp,
                                        rgb_sha256=hashlib.sha256(packet.rgb.tobytes()).hexdigest(),padding=padding))
            entry['groups'][name]=records
            for start in range(0,len(frames),8):
                part=frames[start:start+8];sheet=Image.new('RGB',(1536,624),'#202020');draw=ImageDraw.Draw(sheet)
                for i,packet in enumerate(part):
                    x=(i%4)*384;y=(i//4)*312
                    sheet.paste(Image.fromarray(packet.rgb),(x,y+24));draw.text((x+5,y+5),f'pair {records[start+i]["pair_index"]} / {packet.timestamp:.3f}s',fill='white')
                target=folder/f'{name}-{start//8:02d}.png';sheet.save(target)
                entry['images'].append(dict(path=target.relative_to(OUT).as_posix(),sha256=sha(target),group=name))
        manifest.append(entry)
        print(json.dumps(dict(case=number,pairs_received=pairs,images=len(entry['images'])),ensure_ascii=False),flush=True)
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(cases=6,images=sum(len(e['images']) for e in manifest),GPU=0,video_downloads=0)))

if __name__=='__main__':main()
