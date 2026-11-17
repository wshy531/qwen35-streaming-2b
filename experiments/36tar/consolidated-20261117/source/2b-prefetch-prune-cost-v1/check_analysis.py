"""Check the reporting decision, missing/duplicate detection and mismatch rejection."""
from analyze import summarize
from common import PREP, CASES, read, write

panel=read(PREP/'panel.json')
factors=dict(serial_full=1.,prefetch_full=.9,prefetch_fixed=.81,prefetch_content=.891)
rows=[dict(id=q['id'],case=c,repeat=r,pipeline_wall_seconds=(10+i)*factors[c],reference_match={'generated_ids':True})
      for i,q in enumerate(panel) for c in CASES for r in range(4)]
result=summarize(rows,panel)
fixed=next(x for x in result['comparisons'] if x['candidate']=='prefetch_fixed')
content=next(x for x in result['comparisons'] if x['baseline']=='prefetch_full' and x['candidate']=='prefetch_content')
assert abs(fixed['reduction_fraction']-.1)<1e-12 and fixed['meets_practical_cost_screen']
assert abs(content['reduction_fraction']-.01)<1e-12 and not content['meets_practical_cost_screen']
for damaged in [rows[:-1],rows[:-1]+[rows[0]],[dict(r,reference_match={'generated_ids':False}) if i==0 else r for i,r in enumerate(rows)]]:
    try:summarize(damaged,panel)
    except AssertionError:pass
    else:raise AssertionError('Invalid experiment accepted')
write(PREP/'CPU_analysis.json',dict(status='passed_synthetic_reporting_only',sources=6,rows=96,
      missing_duplicate_mismatch_rejected=True,known_cost_deltas_recovered=True))
print('ANALYSIS_CPU_PASS')
