"""Independent protocol parser/PTS boundary controls, including true EOF and VFR."""
import ast
from fractions import Fraction
import re
import av
import numpy as np
from sember_common import OUT,PREP,read,save
from sember_protocol import parse_choice,prompt,format_ok
from sember_source import CausalSource,PairDriver


def main():
    # Compare the official pure functions, not a hand-written copy of their expected output.
    source=(PREP/'upstream/lmms_eval__tasks__sember__utils.py').read_text(encoding='utf-8')
    nodes=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name in ('_parse_mcq_choice','sember_doc_to_text_mcq')]
    namespace={'re':re,'CHOICE_LABELS':list('ABCDE')}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'pinned_official_pure_functions','exec'),namespace)
    cases=['','A',' a ','AB','DE','B. because','The answer is C.','A or B','No D, choose E','unknown','E','(b)']
    for count in (2,3,4,5):
        for case in cases:assert parse_choice(case,count)==namespace['_parse_mcq_choice'](case,count),(case,count)
    for q in read(PREP/'selection/requests.json'):assert prompt(q)==namespace['sember_doc_to_text_mcq'](q)
    assert format_ok(' B ',5) and not format_ok('Answer: B',5) and parse_choice('Answer: B',5)=='B'
    # A tiny lossless technical fixture; not a capability dataset or a modified natural video.
    path=OUT/'PTS_boundary_fixture.mkv'
    with av.open(str(path),'w') as container:
        stream=container.add_stream('ffv1',rate=12);stream.width=384;stream.height=288;stream.pix_fmt='bgr0'
        for i,seconds in enumerate((0,2,4,6,8,10,12)):
            image=np.full((288,384,3),i*30,dtype=np.uint8)
            frame=av.VideoFrame.from_ndarray(image,format='rgb24');frame.pts=(seconds+100)*12;frame.time_base=Fraction(1,12)
            for packet in stream.encode(frame):container.mux(packet)
        for packet in stream.encode():container.mux(packet)
    submitted=[];source=CausalSource(path)
    def callback(a,b,*,padding):submitted.append((a.timestamp,b.timestamp,padding))
    driver=PairDriver(source,callback)
    try:
        a=driver.advance(.1);assert a['pending_real_frames']==1 and a['pairs']==0
        b=driver.advance(1.999);assert b['pairs']==0 and source.sampled==1
        c=driver.advance(2.);assert c['pairs']==1 and c['last_committed_pts']==2.
        d=driver.advance(4.);assert d['pending_real_frames']==1 and d['last_committed_pts']==2.
        e=driver.advance(12.);assert e['pairs']==4 and e['real_frames']==7 and e['true_EOF_tail_padded']
        f=driver.advance(13.);assert f['pairs']==4 and len(submitted)==4
        assert submitted[-1]==(12.,12.,True)
        assert source.origin==100. and source.eof
        try:driver.advance(1.)
        except ValueError:pass
        else:raise AssertionError('backward question accepted')
    finally:driver.close()
    result=dict(status='passed',official_parser_cases=48,official_prompt_matches=580,
        content_and_format_separate=True,future_real_frame_never_submitted=True,
        pending_held_across_queries=True,true_EOF_pads_once=True,nonzero_source_origin=True,
        gapped_source_PTS=True,backward_query_rejected=True,source_closed=True,
        GPU_jobs=0,pretrained_loads=0,no_capability_claim=True)
    save(OUT/'boundary_preflight.json',result);print(result,flush=True)


if __name__=='__main__':main()
