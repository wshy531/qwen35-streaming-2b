"""Bind a new GPU extension to the unchanged, completed CPU preflight."""
import math
from pathlib import Path

from ..online_memory_2b.isolation import configure, output_path
configure()
from ..online_memory_2b.frozen import REPO
from ..online_memory_2b.protocol import read, sha
from ..long_video_2b.runner import verify as verify_cpu

PROFILE = dict(schema='long-video-external-memory-2b-gpu-v1', CPU_job=217847,
    CPU_protocol_sha256='8be5247cda450094883b8da2e8fcd8931ce1aedc728a54087fbc43dcc54c4517',
    CPU_preflight_sha256='99f8f388252f32e8316baed93461ce910699d879084cfccc343939ccba388b40',
    CPU_source_lock_sha256='0b71c6f7ed3d9c0f0d4b1ffb92cee8fd1fd8b08cd0df5aa842583014205baddf',
    extension_contract_tests=22,
    arms=['M0','M2','M2_A06','LS_shared384_v1','reservoir4','text_only'],
    sources=6,questions=34,sampled_frames=2726,visual_forwards=1360,main_answers=204,
    probe_answers=0,max_new_tokens=8,max_generated_tokens=1632,pretrained_loads=1,
    reset_checks=30,precision='float32',canvas_hw=[192,256],visual_tokens_per_policy=384,
    threshold=0.047536224126815796,worker_seconds=2550,slurm_minutes=45,CPUs=4,RAM_GiB=24,
    minimum_free_GPU_GiB=16,minimum_free_RAM_GiB=12,max_GPU_allocated_GiB=24,max_RSS_GiB=20,
    max_log_bytes=64*2**20,automatic_retry=False,
    semantics='bounded external visual banks; fresh language state for every question',
    independent_holdout=False,independent_semantic_review='pending',
    score_groups=['original28_raw_labels','historical_supported16','historical_ambiguous11',
                  'historical_cutoff1','manual6_source_declared'],
    historical_answer_replay_required=False,
    historical_comparability='same sampled query prefixes; teammate observation core differs')


def extension_lock():
    folder=Path(__file__).parent
    paths=list(folder.rglob('*.py'))+[REPO/'scripts/run_2b_long_video_gpu.py']
    return {p.relative_to(REPO).as_posix():sha(p) for p in sorted(paths)}


def verify_spec(run, spec_path):
    run,spec_path=output_path(run),output_path(spec_path)
    if spec_path!=run/'gpu_protocol.json':
        raise ValueError('GPU protocol must belong to this CPU run')
    spec=read(spec_path)
    if any(spec.get(k)!=v for k,v in PROFILE.items()) or spec.get('run_path')!=str(run):
        raise ValueError('fixed GPU profile changed')
    if spec.get('extension_sha256')!=extension_lock():
        raise ValueError('GPU extension source drift')
    protocol,data=verify_cpu(run)
    if not spec.get('CPU_artifacts_sha256') or 'cpu/preflight.json' not in spec['CPU_artifacts_sha256']:
        raise ValueError('complete approved CPU artifacts required')
    for name,expected in spec['CPU_artifacts_sha256'].items():
        path=output_path(run/name)
        if not path.is_relative_to(run) or sha(path)!=expected:
            raise ValueError('verified CPU artifact changed: '+name)
    proof=read(run/'cpu/preflight.json')
    required={'cpu/preflight.json','cpu/inherited.txt','cpu/actual_query_checks.json'}
    required.update('cpu/'+value['file'] for value in proof['suites'].values())
    if not required.issubset(spec['CPU_artifacts_sha256']):
        raise ValueError('CPU regression and query evidence missing')
    if (proof['status']!='passed' or proof['pretrained_loads']!=0
            or sha(run/'protocol.json')!=spec['CPU_protocol_sha256']
            or sha(run/'cpu/preflight.json')!=spec['CPU_preflight_sha256']
            or sha(run/'source_lock.json')!=spec['CPU_source_lock_sha256']
            or proof['protocol_sha256']!=sha(run/'protocol.json')
            or proof['source_lock_sha256']!=sha(run/'source_lock.json')
            or (proof['sources_RGB_PTS_checked'],proof['sampled_frames'],proof['pair_callbacks'],
                proof['queries_checked'])!=(6,2726,1360,34)):
        raise ValueError('matching successful long-video CPU proof required')
    return spec,protocol,proof,data


class Budget:
    def __init__(self):
        self.answers=self.tokens=0

    def answer(self,result):
        ids=result.get('generated_ids')
        if (not isinstance(result.get('text'),str) or not isinstance(ids,list)
                or not 1<=len(ids)<=8 or any(type(i) is not int or i<0 for i in ids)):
            raise ValueError('invalid result or per-answer budget exceeded')
        if self.answers+1>204 or self.tokens+len(ids)>1632:
            raise RuntimeError('fixed answer/token budget exceeded')
        self.answers+=1
        self.tokens+=len(ids)

    def resources(self,seconds,rss,gpu,log_bytes):
        if (not math.isfinite(seconds) or seconds<0 or seconds>2550
                or not 0<=rss<=20*2**30 or not 0<=gpu<=24*2**30
                or not 0<=log_bytes<=64*2**20):
            raise RuntimeError('worker time, RAM, GPU or log budget exceeded')


def check_query(snapshot,prefix):
    if prefix['query_padding'] is not False:
        raise ValueError('query padding is forbidden')
    expected=dict(received_frames=prefix['arrived_frames'],committed_pairs=prefix['committed_pairs'],
        committed_through_s=prefix['committed_through_s'],observation_age_s=prefix['observation_age_s'],
        pending_frame_id=None if prefix['pending_frame'] is None else prefix['pending_frame']['frame_id'],
        committed_frames_sha256=prefix['historical_committed_frames_sha256'])
    if any(snapshot.get(k)!=v for k,v in expected.items()):
        raise ValueError('GPU query prefix differs from verified CPU snapshot')
