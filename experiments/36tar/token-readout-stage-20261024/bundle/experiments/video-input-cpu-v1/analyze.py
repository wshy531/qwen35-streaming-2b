"""Fixed two-repeat input comparison; costs and identity reported separately."""
from collections import defaultdict
import statistics,platform,sys,time
import av,psutil
from common import ROOT,OUT,read,save,sha


def main():
    complete=read(OUT/'completed.json');assert complete['status']=='complete' and complete['cases']==32
    plan=read(OUT/'inputs/cases.json');groups=read(OUT/'inputs/groups.json')
    reports=[read(OUT/('case-'+str(c['index'])+'.json')) for c in plan]
    assert all(r['case']==c and r['status']=='passed' for c,r in zip(plan,reports))
    lock=read(OUT/'inputs/execution_lock.json')
    for rel,digest in lock['runtime_SHA256'].items():assert sha(ROOT/rel)==digest
    for name,digest in lock['input_SHA256'].items():assert sha(OUT/'inputs'/name)==digest
    stats=[];query_total=samples=0
    for group in groups:
        rows=[r for r in reports if r['case']['group']==group['id']]
        assert len(rows)==4 and all(r['sampled_pixels_and_PTS_all_verified'] and r['all_buffers_released'] for r in rows)
        reference=rows[0]
        for r in rows:
            assert r['query_checks']==reference['query_checks']
            assert r['source']['sampled']==len(group['frames'])
            assert r['source']['decoded_frames']==reference['source']['decoded_frames']
            assert r['source']['peak_application_RGB_buffers']<=3
            assert r['peak_RSS_bytes']<1536*2**20 and r['GPU_jobs']==r['pretrained_loads']==0
            query_total+=r['query_count'];samples+=r['source']['sampled']
        costs={}
        for config in ('slice1','auto4'):
            rs=[r for r in rows if r['case']['config']==config];assert len(rs)==2
            costs[config]=dict(
                median_input_wall_seconds=statistics.median(r['input_wall_seconds'] for r in rs),
                all_input_wall_seconds=[r['input_wall_seconds'] for r in rs],
                median_decode_seconds=statistics.median(r['source']['decode_seconds'] for r in rs),
                median_transform_seconds=statistics.median(r['source']['transform_seconds'] for r in rs),
                median_first_sample_seconds=statistics.median(r['source']['first_sample_seconds'] for r in rs),
                max_peak_RSS_bytes=max(r['peak_RSS_bytes'] for r in rs),
                max_application_RGB_buffers=max(r['source']['peak_application_RGB_buffers'] for r in rs),
                max_process_threads=max(r['max_observed_threads'] for r in rs))
        stats.append(dict(group=group['id'],cohort=group['cohort'],samples=len(group['frames']),
            decoded_frames_per_run=reference['source']['decoded_frames'],
            codec=reference['source']['codec']['codec'],
            source_width=reference['source']['codec']['width'],source_height=reference['source']['codec']['height'],
            query_count=reference['query_count'],costs=costs,
            input_wall_reduction=1-costs['auto4']['median_input_wall_seconds']/costs['slice1']['median_input_wall_seconds']))
    summary={}
    for label,subset in [('all8',stats),('long6',[r for r in stats if r['cohort']=='stream34']),('short2',[r for r in stats if r['cohort']!='stream34'])]:
        base=sum(r['costs']['slice1']['median_input_wall_seconds'] for r in subset)
        parallel=sum(r['costs']['auto4']['median_input_wall_seconds'] for r in subset)
        summary[label]=dict(sources=len(subset),slice1_seconds=base,auto4_seconds=parallel,
            input_wall_reduction=1-parallel/base,speedup=base/parallel,
            median_first_sample_seconds={c:statistics.median(r['costs'][c]['median_first_sample_seconds'] for r in subset) for c in ('slice1','auto4')},
            max_peak_RSS_bytes={c:max(r['costs'][c]['max_peak_RSS_bytes'] for r in subset) for c in ('slice1','auto4')})
    report=dict(status='passed',cases=32,sampled_pixel_timestamp_checks=samples,query_checks=query_total,
        all_fixed_long34_reproduced_per_config_repeat=True,groups=stats,summary=summary,
        environment=dict(Python=sys.version,platform=platform.platform(),processor=platform.processor(),
            logical_cpus=psutil.cpu_count(),PyAV=av.__version__,cuda_disabled=True,
            monotonic_clock=dict(implementation=time.get_clock_info('monotonic').implementation,
                                 resolution_seconds=time.get_clock_info('monotonic').resolution)),
        conclusions=['Pixel and causal identity is input evidence, not a new model capability test.',
                     'Four decoder threads use more CPU than one; this is input engineering, not a free algorithmic gain.',
                     'Warm file cache, two repeats, eight sources, local hardware only.',
                     'Application RGB count does not include internal codec DPB/thread buffers.',
                     'Short first-sample timings are limited by the recorded monotonic-clock resolution.',
                     'No end-to-end2B or edge/paced1x speed was measured.'])
    save(OUT/'answer_and_cost_audit.json',report)
    print(summary)


if __name__=='__main__':main()
