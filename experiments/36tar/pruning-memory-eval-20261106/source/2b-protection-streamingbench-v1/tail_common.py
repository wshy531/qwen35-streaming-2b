from fair_common import *
T=R/'runs/protection_stream_tail_prepare_20261103_01'

def verify_tail():
    verify()
    for n,h in read(T/'execution_lock.json').items():
        assert sha(R/n)==h,n
    m=read(T/'manifest.json')
    assert sha(R/'runs'/m['parent']/'answers.jsonl')==m['parent_answers_sha256']
    assert read(T/'CPU_local.json')==read(T/'CPU_cluster.json')
