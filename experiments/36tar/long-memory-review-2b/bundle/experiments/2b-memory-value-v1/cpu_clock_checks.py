"""Official clock function isolated from source; gap and query gate contracts."""
import ast,hashlib,json
from pathlib import Path
from fractions import Fraction
from clock_protocol import clock_seconds,PairGate
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/2b_memory_value_prepare_20261021_01'


def main():
    source=OUT/'author_sources/EgoRAG__egorag__utils__util.py'
    parsed=ast.parse(source.read_text(encoding='utf-8'))
    function=next(n for n in parsed.body if isinstance(n,ast.FunctionDef) and n.name=='time_to_frame_idx')
    # Compile only this small arithmetic function; do not import API/client code.
    namespace={};exec(compile(ast.Module(body=[function],type_ignores=[]),'official-clock-function-only','exec'),namespace)
    official=namespace['time_to_frame_idx'];n=0
    for rate in (20,25,30):
        for text in ('11094208','11100000','17210116','22053000'):
            assert clock_seconds(text,rate)==Fraction(official(text,rate),rate);n+=1
    gate=PairGate();assert gate.arrive(0,0.) is None
    assert gate.query(.5)['committed_pairs']==0
    assert gate.arrive(1,1.)==((0,0.),(1,1.))
    assert gate.arrive(2,2.) is None
    assert gate.arrive(3,10803.) is None and gate.discarded_pending==1
    assert gate.query(10803.)['committed_through_s']==1.
    assert gate.arrive(4,10804.)==((3,10803.),(4,10804.))
    before=(gate.pairs,gate.pending,gate.last_seen)
    for action in (lambda:gate.arrive(4,10805.),lambda:gate.query(10000.)):
        try:action()
        except ValueError:pass
        else:raise AssertionError('duplicate/past query not rejected')
        assert (gate.pairs,gate.pending,gate.last_seen)==before
    assert gate.arrive(5,10805.) is None;gate.end();assert gate.pending is None and gate.discarded_pending==2
    for bad in ('11096000','25000000','11094299','12471319DAY1_12474208'):
        try:clock_seconds(bad,20)
        except ValueError:pass
        else:raise AssertionError('invalid source clock accepted')
    result=dict(status='passed',official_clock_equal_cases=n,recording_gap_not_paired=True,
        question_does_not_read_pending_or_future=True,odd_tail_discarded=True,duplicate_and_past_query_rejected=True,
        official_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        clock_format='HHMMSSFF, FF is frame index at actual media fps, not centiseconds',
        limitations='only arithmetic/scheduler contracts; actual media FPS/PTS and question evidence still pending',GPU_jobs=0)
    with (OUT/'CPU_clock_gate.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(result)


if __name__=='__main__':main()
