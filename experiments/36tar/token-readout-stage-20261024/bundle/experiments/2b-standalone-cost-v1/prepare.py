"""Freeze old303 inputs and balanced blocks, without inspecting answers."""
from collections import defaultdict,Counter
from pathlib import Path
import json,shutil
from common import ROOT,OUT,ARMS,read,save,sha


def main():
    source=ROOT/'runs/2b_matched_coverage_prepare_20261023_01'
    OUT.mkdir(exist_ok=False);(OUT/'inputs').mkdir()
    for name in ('groups.json','questions.json','snapshots34.json'):
        shutil.copyfile(source/'inputs'/name,OUT/'inputs'/name)
    shutil.copyfile(source/'labels_LOCAL_ONLY.json',OUT/'labels_LOCAL_ONLY.json')
    groups=read(OUT/'inputs/groups.json');questions=read(OUT/'inputs/questions.json')
    assert len(groups)==len({g['sha256'] for g in groups})==112
    cohorts=defaultdict(list)
    for group in groups:cohorts[group['cohort']].append(group)
    assigned={}
    for cohort,gs in cohorts.items():
        for index,group in enumerate(sorted(gs,key=lambda g:g['sha256'])):assigned[group['id']]=index%3
    jobs=[]
    for block in range(3):
        members=[g['id'] for g in groups if assigned[g['id']]==block]
        order=ARMS[block:]+ARMS[:block]
        for arm in order:
            jobs.append(dict(block=block,arm=arm,groups=members,
                             questions=sum(q['group'] in members for q in questions)))
    assert sum(job['questions'] for job in jobs)==909 and len(jobs)==9
    warmup=next(g for g in sorted(groups,key=lambda g:g['sha256']) if len(g['frames'])>=16)
    save(OUT/'inputs/jobs.json',jobs)
    save(OUT/'inputs/protocol.json',dict(arms=list(ARMS),questions=303,measured_answers=909,
        warmup_answers=9,answers_max=918,generated_tokens_max=7344,measured_visual_encodings=11943,
        warmup_visual_encodings=72,visual_encodings=12015,pretrained_loads=9,
        warmup_group=warmup['id'],warmup_frames=16,context_cap=2048,visual_budget_max=384,
        max_new_tokens=8,max_nodes=8,dtype='FP32',model_revision='15852e8c16360a2fea060d615a32b45270f8a8fc',
        independent_processes=9,one_active_bank=True,only_change_keeps_previous_pair=True,
        balanced_order='3 cohort-stratified SHA blocks; Latin rotation of arm order',
        no_main_language_cache=True,no_full_history_archive=True,wallclock_1x=False,
        deployment_device='A800 cluster, not an edge device',
        stop='one frozen workload; report component and full wall time, quality regressions, no parameter sweep',
        clock='perf_counter/monotonic; explicit CUDA synchronization at vision/QA stage boundaries',
        inference_wall_includes='decoding, transform, validation, encoding, transfer, selection, memory, isolated QA, audit, logs',
        integrity_and_audit_overhead_reported_separately=True,
        input_sha256={n:sha(OUT/'inputs'/n) for n in ('groups.json','questions.json','snapshots34.json','jobs.json')}))
    reference=[]
    for line in (source/'results-218063/answers.jsonl').read_text(encoding='utf-8').splitlines():
        r=json.loads(line)
        if r['arm'] in ARMS:
            reference.append({k:r[k] for k in ('id','arm','generated_ids','input_ids_sha256','node_digest')})
    assert len(reference)==909
    save(OUT/'historical_references_LOCAL_ONLY.json',dict(records=reference,source_SHA256=sha(source/'results-218063/answers.jsonl')))
    here=Path(__file__).parent
    old=(ROOT/'experiments/2b-matched-coverage-v1/session.py').read_text(encoding='utf-8')
    old=old.replace('def __init__(self, env, model, builder):','def __init__(self, env, model, builder, arm):')
    old=old.replace('self.bank = Banks()','self.arm=arm\n        self.bank = Banks(arm)\n        self.audit_seconds=0.')
    old=old.replace('if arm not in ARMS or','if arm != self.arm or')
    old=old.replace('before, digest = self.status(), self.bank.fingerprint()',
                    'audit_started=time.monotonic()\n        before, digest = self.status(), self.bank.fingerprint()')
    old=old.replace('selector_before=self.bank.selector_digest()',
                    'selector_before=self.bank.selector_digest()\n        self.audit_seconds+=time.monotonic()-audit_started')
    old=old.replace('assert self.status() == before and self.bank.fingerprint() == digest',
                    'audit_started=time.monotonic()\n        assert self.status() == before and self.bank.fingerprint() == digest')
    old=old.replace('assert self.bank.selector_digest()==selector_before',
                    'assert self.bank.selector_digest()==selector_before\n        self.audit_seconds+=time.monotonic()-audit_started')
    first=old.index('    def ask(self, question, *, read_from, on_token):')
    last=old.index('    def close(self):',first)
    old=old[:first]+'''    def ask(self, question, *, read_from, on_token):
        if read_from != 'standalone':raise ValueError('explicit standalone read required')
        answer=self.ask_arm(self.arm,question,lambda t:on_token(dict(arm=self.arm,**t)))
        self.queries+=1
        return dict(arms={self.arm:answer},order=[self.arm],all_QA_branches_released=True)

'''+old[last:]
    old=old.replace('matched-coverage-qa','standalone-cost-qa')
    (here/'session.py').write_text(old,encoding='utf-8')
    print(dict(status='prepared',jobs=len(jobs),measured_answers=909,warmup_source=warmup['id']))


if __name__=='__main__':main()
