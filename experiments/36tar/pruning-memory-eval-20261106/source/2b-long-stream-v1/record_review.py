"""Single-assistant content record after actual sheet inspection.

Preserves all28 official questions. Six separately named manual development
questions add delayed temporal evidence; they are not extra independent videos.
"""
from collections import Counter
import hashlib
import json

from inventory_local import ROOT,OUT


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(name,value):
    with (OUT/name).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2)


def main():
    original=read(OUT/'existing6_EPM28_candidates_LOCAL_ONLY.json')
    traces=read(OUT/'actual_sampling_all6.json')['videos']
    # Actual pages viewed this session; overview is not full1FPS human review.
    dense=[[8,9,10,11,13,14,16,17,18],[5,14],[7,15],[0,2,11,13,15,19],[2,9,10,14,15,17],[0,8]]
    seen={}
    for trace,indices in zip(traces,dense):
        pages=trace['overview_pages']+[trace['pages'][i] for i in indices]
        for p in pages:
            assert hashlib.sha256((ROOT/p['path']).read_bytes()).hexdigest()==p['sha256']
        seen[trace['source_path']]=pages
    # verdict, witness samples, reason, late-cue/necessity limitation
    notes={
      70:('ambiguous',[], 'Specific spatula pickup/location not established in viewed prefix pages.', 'Do not infer cabinet source from generic kitchen layout.'),
      71:('ambiguous',[], 'Multiple transfers visible; exactly four scoops not independently established by this review.', 'Do not rewrite official count or infer it from the answer.'),
      72:('supported',[405,408], 'Pink towel used on hands near sink before411s.', 'Near-query evidence; not long-memory necessity.'),
      73:('supported',[318,322,325], 'Drawer opens and a dark serving spoon is taken and used on food.', 'Spoon remains in kitchen context later; witness age is not necessity.'),
      104:('ambiguous',[], 'Toy-rack and shelf options overlap; exact alien-box identity unresolved at input resolution.', 'No label correction or model-conditioned exclusion.'),
      105:('ambiguous',[], 'Shelf versus toy-rack distinction and exact robot identity not sufficiently established.', 'Closely related location options retained.'),
      102:('supported',[339,340,341], 'Metal shopping cart is adjacent to Duracell battery display.', 'Direct recent evidence despite a query after5min.'),
      103:('supported',[395,430], 'Blue shopping basket visible while moving through store.', 'Basket is repeatedly visible, not necessary early memory.'),
      106:('ambiguous',[], 'Floor and ground options overlap for the container placement.', 'Physical position can be seen but unique MC answer is ambiguous.'),
      107:('ambiguous',[], 'Repeated mortar scooping; exact five count not independently verified here.', 'Requires separate count audit, not declared official-label error.'),
      108:('supported',[200,300,320], 'Grey mortar/cement is transferred from bucket during panel plastering.', 'Material inferred with construction context; repeats near query.'),
      109:('ambiguous',[], 'Exact scraper source versus adjacent cabinet/workbench not resolved.', 'No automatic cabinet answer from room context.'),
      130:('ambiguous',[], 'Discarded package/food identity cannot be verified as beef sauce at current canvas.', 'Do not claim incorrect official annotation.'),
      131:('supported',[273,275,277], 'Onion is removed from clear plastic bag.', 'Direct recent transition.'),
      68:('supported',[273,274,275], 'Red round floor cushion visible in furnished living area beyond kitchen.', 'Room category uses visible context, not full floorplan knowledge.'),
      69:('supported',[273,275,277], 'The onion is held within and removed from the clear bag.', 'Same event as131, correlated question rather than independent event.'),
      67:('supported',[330,331], 'Long wrap carton is on the refrigerator door shelf before retrieval.', 'Retrieval occurs later; location itself is visible before committed cutoff.'),
      66:('ambiguous',[], 'At refrigerator versus inside refrigerator overlap; placement completion is also near query.', 'Keep original alternatives; no scoring repair.'),
      132:('supported',[371,372,373], 'Onion is carried into the open fridge/door-shelf area.', 'Release appears later; only object entering the fridge is claimed.'),
      133:('not_established_at_committed_cutoff',[], 'Pan lid retrieval becomes clear around472-473s, while paired observation at query472 ends471s.', 'Do not use future frames or shift original query; report pairing/timing limitation.'),
      180:('ambiguous',[], 'Drawer is visible but exact blue screwdriver identity is unresolved at59s.', 'Location plausibility alone is insufficient.'),
      181:('supported',[241,242,243], 'Black basin is lowered onto blue drum and released.', 'Near-query placement, not old evidence.'),
      182:('supported',[357,358,359], 'Metal tool is accessed from opened tool drawer before moving to mower.', 'Fine tool name not separately certified; drawer location is visible.'),
      183:('supported',[408,409,410], 'Wrench is selected among wrenches in the open drawer.', 'Current/very recent evidence.'),
      144:('supported',[125,130], 'Nearby sedan in parking area is white.', 'Several white vehicles occur; target follows local query context.'),
      145:('supported',[192,193], 'Blue trash bins are beside entrance doors.', 'Committed cutoff193s; pending194s frame is not used.'),
      147:('ambiguous',[], 'Beside trashcan and beside dustbin are synonymous options.', 'Duplicate semantic answers retained, not forced into strict unique choice.'),
      146:('supported',[208,209,210], 'Wearer looks down and black upper clothing is visible.', 'Direct near-query view.'),
    }
    by_source={t['source_path']:t for t in traces}

    def witnesses(source,times,cutoff):
        trace=by_source[source];result=[]
        visible_ids={f['frame_id'] for p in seen[source] for f in p['frames']}
        for t in times:
            item=min(trace['frames'],key=lambda f:abs(f['pts_s']-t))
            assert abs(item['pts_s']-t)<.05 and item['pts_s']<=cutoff
            assert item['frame_id'] in visible_ids,'witness image must have been viewed'
            result.append(item)
        return result

    ledger=[]
    for row in original:
        number=row['original']['id'];v,ts,reason,limit=notes[number]
        trace=by_source[row['source_path']]
        snap=next(q for q in trace['query_snapshots'] if q['query_id']==row['id'])
        ws=witnesses(row['source_path'],ts,snap['committed_through_s'])
        ledger.append(dict(id=row['id'],source_path=row['source_path'],original=row['original'],
            verdict=v,witnesses=ws,reason=reason,limitations=limit,
            query_snapshot=snap,witness_gap_s=None if not ws else row['query_time_s']-max(w['pts_s'] for w in ws),
            old_evidence_necessity_proven=False,model_outputs_used=False,
            reviewer='single_assistant_overview_and_listed_dense_pages',second_review=False))
    save('official28_content_review.json',ledger)
    # All manual examples are development diagnostics, not an unseen benchmark.
    manual_specs=[
      (0,430.,'Which happened first: taking plastic wrap from the upper cupboard or opening the refrigerator?',
       ['Opening the refrigerator before taking the wrap','Neither action occurred',
        'Taking the wrap before opening the refrigerator','Only opening the refrigerator occurred'],2,
       [206,209,211,226,228], 'Wrap taken from upper cupboard and used before refrigerator opens.',
       'Two earlier transitions; all overview pages seen, but necessity still needs intervention.'),
      (1,470.,'Which order did the wearer follow in the store?',
       ['Inspecting boxed toys before passing the Duracell battery display',
        'Passing the Duracell display before inspecting boxed toys',
        'Inspecting boxed toys without ever passing a battery display',
        'Passing a battery display without ever inspecting boxed toys'],0,
       [45,50,339,341], 'Toy inspection precedes the later battery-display/cart scene.',
       'Box identity is generic, not the ambiguous alien/robot label.'),
      (2,450.,'Which sequence of activities was shown earlier in this clip?',
       ['Applying mortar to a panel, handling blue buckets, using the orange mixer',
        'Using the orange mixer, applying mortar to a panel, handling blue buckets',
        'Handling blue buckets, using the orange mixer, applying mortar to a panel',
        'Using the orange mixer, handling blue buckets, applying mortar to a panel'],3,
       [20,115,200], 'Orange mixer work precedes blue-bucket handling and subsequent panel application.',
       'Tool/workflow context could help guessing; keep text-only control and do not claim necessity.'),
      (3,460.,'After cutting the yellow pepper into strips, which ingredient was cut into strips next?',
       ['Cabbage','Green pepper','Onion','Carrot'],1,
       [20,48,54,60,70], 'Yellow pepper strips precede green pepper strips; cabbage is later.',
       'Vegetables remain visible later, but their order requires earlier sequence evidence.'),
      (4,440.,'What color was the first cloth taken out of the tool drawer and used for wiping?',
       ['Red','Blue','Green','White'],0,
       [60,62,65,70], 'Red cloth is taken from drawer and used before the later blue cloth.',
       'Scene has many red objects; answer dependence must be checked, not inferred from gap.'),
      (5,450.,'At the initial road crossing, what small roadside fixture stood beside the pedestrian button pole?',
       ['Parking meter','Bicycle rack','Fire hydrant','Public trash bin'],2,
       [4,5,7], 'Red fire hydrant is next to the pedestrian-crossing pole before entering the store.',
       'Small object at this canvas; cross-source reuse and text priors need controls.'),
    ]
    manual=[]
    for i,(source_index,time_s,question,options,truth,times,reason,limit) in enumerate(manual_specs):
        trace=traces[source_index]
        prefix=[f for f in trace['frames'] if f['pts_s']<=time_s]
        prefix=prefix[:len(prefix)//2*2]
        ws=witnesses(trace['source_path'],times,prefix[-1]['pts_s'])
        gap=time_s-max(w['pts_s'] for w in ws)
        assert gap>=120
        manual.append(dict(id=f'local_longstream_v1:{i+1:02d}',source_path=trace['source_path'],
            question=question,options=options,answer_index=truth,query_time_s=time_s,
            annotation_origin='new_manual_single_assistant_NOT_official_OVO',
            split='development_integration_NOT_unseen_holdout',
            witnesses=ws,witness_gap_s=gap,reason=reason,limitations=limit,
            verdict='single_reviewer_supported_candidate',independent_second_review=False,
            old_evidence_necessity_proven=False,model_outputs_used=False,
            original_OVO_questions_unchanged=True))
    save('manual6_delayed_questions_LOCAL_ONLY.json',manual)
    summary=dict(status='single_review_complete_preserving_all_original_questions',
        original_questions=28,original_verdicts=dict(Counter(q['verdict'] for q in ledger)),
        manual_delayed_candidates=6,manual_witness_gap_range_s=[min(q['witness_gap_s'] for q in manual),max(q['witness_gap_s'] for q in manual)],
        manual_are_not_extra_independent_sources=True,independent_second_review=False,
        actual_viewed_pages={k:[dict(path=p['path'],sha256=p['sha256']) for p in v] for k,v in seen.items()},
        full1FPS_pages_all_human_viewed=False,method='all25 overview pages plus explicitly listed dense pages',
        model_loads=0,GPU_jobs=0,formal_long_benchmark_complete=False)
    save('content_review_summary.json',summary)
    print({k:v for k,v in summary.items() if k!='actual_viewed_pages'})


if __name__=='__main__':main()
