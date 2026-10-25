"""Lightweight checks in the existing local environment; never fake torch tests."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from ..online_memory_2b.isolation import configure,output_path
from ..online_memory_2b.protocol import save,verify_package
from ..online_memory_2b.receipts import successful_counts
from ..online_memory_2b.frozen import REPO
from .controls import PRESETS,canvas,schedule
from ..sember_input_diagnostic.controls import canvas as baseline


def run(out):
    root=configure(); out=output_path(out); out.mkdir(exist_ok=False)
    results=[]
    try:
        for name,count in [('sember_102_2b',11),('sember_readout_diagnostic',17),
                           ('sember_input_diagnostic',13),('sember_resolution_2b',20)]:
            logfile=out/(name+'.txt')
            with logfile.open('x',encoding='utf-8') as stream:
                subprocess.run([sys.executable,'-B','-m','unittest','discover','-s',
                    str(REPO/'experiments/wshy'/name/'tests'),'-v'],cwd=REPO,
                    stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=180)
            successful_counts(logfile.read_text(encoding='utf-8'),[count])
            results.append(dict(suite=name,passed=count))
        from .runner import history,code_lock
        _,records,reused=history(); verify_package()
        paths=list(Path(__file__).parent.rglob('*.py'))+[REPO/'scripts/run_2b_sember_resolution.py']
        for p in paths:
            text=p.read_text(encoding='utf-8'); ast.parse(text,filename=str(p))
            if any(line.rstrip()!=line for line in text.splitlines()): raise ValueError('trailing whitespace: '+str(p))
        import av
        real=[]
        video_dir=output_path(root/'data/S-EMBER/897886cc967b2d0e6e45f1200605f5d3b0ffbb23/videos')
        for path in sorted(video_dir.glob('*.mp4')):
            output_path(path)
            with av.open(str(path)) as source:
                for i,frame in enumerate(source.decode(video=0)):
                    if i>=3: break
                    hashes={}
                    for resolution,hw in PRESETS.items():
                        if resolution!='high' and canvas(frame,resolution).tobytes()!=baseline(frame,*hw).tobytes():
                            raise ValueError('real RGB endpoint drift')
                        if canvas(frame,resolution).size!=tuple(reversed(hw)):
                            raise ValueError('canvas dimensions')
                        hashes[resolution]=hashlib.sha256(canvas(frame,resolution).tobytes()).hexdigest()
                    real.append(dict(file=path.name,frame=i,pts=str(frame.pts),
                        rgb_sha256=hashes))
        if len(real)!=12: raise ValueError('four existing local videos, three frames each required')
        save(out/'results.json',dict(status='local_checks_passed',tests=results,AST_files=len(paths),
            frozen_SHA_entries=156,historical_reused_answers=len(reused),schedule=schedule([r['query'] for r in records]),
            real_RGB_checks=real,source_lock=code_lock(),
            dependency_availability={k:importlib.util.find_spec(k) is not None for k in ('torch','transformers','numpy','PIL','av')},
            torch_numeric_tests='not_run_in_lightweight_check; mandatory cluster preflight',
            cluster_connections=0,real_model_loads=0,jobs_submitted=0))
        print(json.dumps(dict(status='passed',local_tests=61,real_frames=12,reused_answers=1530,out=str(out))))
    except BaseException as exc:
        save(out/'failed.json',dict(status='failed',exception=repr(exc)))
        raise
