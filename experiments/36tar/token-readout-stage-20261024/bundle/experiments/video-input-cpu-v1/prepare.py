"""Precommit8 sources,32 cases and canonical query schedules without model scores."""
from common import ROOT,OUT,CONFIGS,read,save,sha


def main():
    OUT.mkdir(exist_ok=False);(OUT/'inputs').mkdir()
    parent=ROOT/'runs/2b_standalone_cost_prepare_20261023_01/inputs'
    all_groups=read(parent/'groups.json');all_questions=read(parent/'questions.json')
    chosen=[g for g in all_groups if g['cohort']=='stream34']
    chosen += [min((g for g in all_groups if g['cohort']==c),key=lambda g:g['sha256']) for c in ('natural171','T98')]
    chosen.sort(key=lambda g:g['sha256']);assert len(chosen)==8
    ids={g['id'] for g in chosen};questions=[q for q in all_questions if q['group'] in ids]
    save(OUT/'inputs/groups.json',chosen);save(OUT/'inputs/questions.json',questions)
    save(OUT/'inputs/snapshots34.json',read(parent/'snapshots34.json'))
    cases=[]
    for number,group in enumerate(chosen):
        order=('slice1','auto4') if number%2==0 else ('auto4','slice1')
        for repeat in range(2):
            for config in order if repeat==0 else order[::-1]:
                cases.append(dict(index=len(cases),group=group['id'],repeat=repeat,config=config))
    save(OUT/'inputs/cases.json',cases)
    save(OUT/'inputs/protocol.json',dict(configs=CONFIGS,groups=8,long_sources=6,short_sources=2,
        planned_repeats=2,cases=32,case_seconds=240,total_seconds=1800,max_RSS_bytes=1536*2**20,
        max_log_bytes=8*2**20,max_application_RGB_buffers=3,GPU_jobs=0,pretrained_loads=0,
        execution='local CPU, sequential fresh child processes',
        first_sample_metric='after media SHA check, from opening/decoding start to first sampled RGB',
        media_SHA_before_decode_warms_OS_file_cache=True,
        no_full_RGB_archive=True,do_not_claim_internal_codec_buffer_is3=True,
        fixed_question_count=len(questions),decord_policy='only if already installed; no install or download',
        stop='one fixed paired run; any pixel/causal/buffer mismatch stops candidate; no sampling or conversion tuning',
        future_validation='decoder config only; no inference quality/speed or wallclock1x claim',
        input_SHA256={p.name:sha(p) for p in (OUT/'inputs').iterdir() if p.is_file()}))
    print(dict(status='prepared',sources=8,cases=len(cases),questions=len(questions)))


if __name__=='__main__':main()
