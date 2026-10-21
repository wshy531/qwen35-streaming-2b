import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from contextlib import ExitStack,nullcontext
from types import SimpleNamespace

from experiments.wshy.online_memory_2b.isolation import output_path
from experiments.wshy.online_memory_2b.protocol import read,sha
from experiments.wshy.long_video_gpu_2b.gates import PROFILE,Budget,check_query,extension_lock,verify_spec
from experiments.wshy.long_video_gpu_2b.scoring import build,scorer

ROOT=Path('D:/AI_project/projects/streaming_video_wsy' if __import__('os').name=='nt' else '/home/share/czy/wsy')


class BudgetContracts(unittest.TestCase):
    def test_exact_maximum_grid_and_token_budget_is_allowed(self):
        budget=Budget()
        for _ in range(204):
            budget.answer(dict(text='A',generated_ids=[1]*8))
        self.assertEqual((budget.answers,budget.tokens),(204,1632))
        with self.assertRaises(RuntimeError):
            budget.answer(dict(text='A',generated_ids=[1]))
        self.assertEqual((budget.answers,budget.tokens),(204,1632))

    def test_empty_excess_invalid_tokens_are_rejected_without_advancing(self):
        for ids in ([],[1]*9,[-1],[True],['1'],None):
            budget=Budget()
            with self.subTest(ids=ids),self.assertRaises(ValueError):
                budget.answer(dict(text='A',generated_ids=ids))
            self.assertEqual((budget.answers,budget.tokens),(0,0))

    def test_nontext_result_is_rejected(self):
        with self.assertRaises(ValueError):
            Budget().answer(dict(text=None,generated_ids=[1]))

    def test_exact_resource_bounds_are_allowed(self):
        Budget().resources(2550,20*2**30,24*2**30,64*2**20)

    def test_each_excess_resource_and_nonfinite_time_is_rejected(self):
        for values in ((2551,0,0,0),(0,20*2**30+1,0,0),(0,0,24*2**30+1,0),
                       (0,0,0,64*2**20+1),(float('nan'),0,0,0),(-1,0,0,0)):
            with self.subTest(values=values),self.assertRaises(RuntimeError):
                Budget().resources(*values)


class QueryGateContracts(unittest.TestCase):
    def fixture(self):
        prefix=dict(arrived_frames=5,committed_pairs=2,committed_through_s=3.,observation_age_s=1.,
            pending_frame=dict(frame_id=4),historical_committed_frames_sha256='a'*64,query_padding=False)
        snap=dict(received_frames=5,committed_pairs=2,committed_through_s=3.,observation_age_s=1.,
            pending_frame_id=4,committed_frames_sha256='a'*64)
        return snap,prefix

    def test_verified_query_prefix_is_accepted(self):
        snap,prefix=self.fixture()
        check_query(snap,prefix)

    def test_changed_pair_pending_frame_hash_or_padding_is_rejected(self):
        snap,prefix=self.fixture()
        for key,value in (('committed_pairs',3),('pending_frame',None),
                          ('historical_committed_frames_sha256','b'*64),('query_padding',True)):
            with self.subTest(key=key),self.assertRaises(ValueError):
                check_query(snap,dict(prefix,**{key:value}))


class SpecContracts(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run=output_path(Path(self.temp.name)/'run')
        (self.run/'cpu').mkdir(parents=True)
        # A gate fixture replaces CPU verification and pins fixture hashes.
        # Real canonical CPU identity is tested separately and remains PROFILE.
        self.cpu=dict(status='passed',pretrained_loads=0,sources_RGB_PTS_checked=6,sampled_frames=2726,
            pair_callbacks=1360,queries_checked=34,suites={'tests':dict(file='tests.txt')})
        for name,content in (('protocol.json','{}'),('source_lock.json','{}'),
                             ('cpu/inherited.txt','proof'),('cpu/actual_query_checks.json','[]'),('cpu/tests.txt','OK')):
            (self.run/name).write_text(content,encoding='utf-8')
        self.cpu.update(protocol_sha256=sha(self.run/'protocol.json'),source_lock_sha256=sha(self.run/'source_lock.json'))
        (self.run/'cpu/preflight.json').write_text(json.dumps(self.cpu),encoding='utf-8')
        self.profile=dict(PROFILE,CPU_protocol_sha256=sha(self.run/'protocol.json'),
            CPU_preflight_sha256=sha(self.run/'cpu/preflight.json'),CPU_source_lock_sha256=sha(self.run/'source_lock.json'))
        self.spec=dict(self.profile,run_path=str(self.run),extension_sha256=extension_lock(),
            CPU_artifacts_sha256={p.relative_to(self.run).as_posix():sha(p) for p in self.run.rglob('*') if p.is_file()})
        self.path=self.run/'gpu_protocol.json'
        self.p1=patch('experiments.wshy.long_video_gpu_2b.gates.PROFILE',self.profile)
        self.p2=patch('experiments.wshy.long_video_gpu_2b.gates.verify_cpu',return_value=({},([],[],[])))
        self.p1.start();self.addCleanup(self.p1.stop)
        self.p2.start();self.addCleanup(self.p2.stop)

    def write(self):
        self.path.write_text(json.dumps(self.spec),encoding='utf-8')

    def test_complete_cpu_bound_extension_is_accepted(self):
        self.write()
        self.assertEqual(verify_spec(self.run,self.path)[0]['main_answers'],204)

    def test_changed_arm_budget_or_cpu_identity_is_rejected(self):
        for key,value in (('main_answers',205),('CPU_job',1),('CPU_preflight_sha256','0'*64),('arms',['M0'])):
            spec=copy.deepcopy(self.spec)
            spec[key]=value
            self.path.write_text(json.dumps(spec),encoding='utf-8')
            with self.subTest(key=key),self.assertRaises(ValueError):
                verify_spec(self.run,self.path)

    def test_changed_extension_or_missing_regression_log_is_rejected(self):
        self.spec['extension_sha256']={}
        self.write()
        with self.assertRaises(ValueError):
            verify_spec(self.run,self.path)
        self.spec['extension_sha256']=extension_lock()
        del self.spec['CPU_artifacts_sha256']['cpu/tests.txt']
        self.write()
        with self.assertRaises(ValueError):
            verify_spec(self.run,self.path)

    def test_modified_cpu_proof_even_with_recomputed_artifact_hash_is_rejected(self):
        p=self.run/'cpu/preflight.json'
        p.write_text(json.dumps(dict(self.cpu,pretrained_loads=1)),encoding='utf-8')
        self.spec['CPU_artifacts_sha256']['cpu/preflight.json']=sha(p)
        self.write()
        with self.assertRaises(ValueError):
            verify_spec(self.run,self.path)

    def test_external_protocol_path_is_rejected(self):
        self.write()
        with self.assertRaises(ValueError):
            verify_spec(self.run,self.run/'another_protocol.json')


class ScoringContracts(unittest.TestCase):
    def fixture(self):
        queries=[];catalog=[];predictions=[]
        for i in range(34):
            original=i<28
            qid=('ovo:'+str(i) if original else 'local_longstream_v1:'+str(i-27).zfill(2))
            admission=('supported' if i<16 else 'ambiguous' if i<27 else
                       'not_established_at_committed_cutoff' if i==27 else 'manual_single_reviewer_candidate')
            queries.append(dict(id=qid,source_path='source'+str(i%6),query_time_s=100,
                question='Which?',options=['first','second'],cohort='original' if original else 'manual'))
            catalog.append(dict(id=qid,label_index=0,label_origin='exactly_matched_official_metadata' if original
                else 'single_assistant_manual_source_declaration',declared_admission=admission,
                independent_review='pending',confirmed_accuracy_eligible=False))
            for arm in PROFILE['arms']:
                predictions.append(dict(id=qid,arm=arm,status='ok',text='A'))
        return queries,catalog,predictions

    def build(self,fixture):
        return build(*fixture,scorer().measure)

    def test_complete_groups_keep_expected_denominators(self):
        summary,details,review=self.build(self.fixture())
        self.assertEqual([v['M0']['questions'] for v in summary['tables'].values()],[28,16,11,1,6])
        self.assertEqual(len(details),204)
        self.assertIsNone(summary['confirmed_accuracy'])
        self.assertEqual(summary['confirmed_accuracy_denominator'],0)

    def test_missing_or_duplicate_answer_and_missing_label_are_rejected(self):
        for mode in ('missing','duplicate','label'):
            q,c,p=self.fixture()
            if mode=='missing':p.pop()
            elif mode=='duplicate':p[-1]=dict(p[0])
            else:c.pop()
            with self.subTest(mode=mode),self.assertRaises(ValueError):
                self.build((q,c,p))

    def test_invalid_truth_or_changed_admission_cannot_silently_change_denominator(self):
        for key,value in (('label_index',None),('label_index',True),('declared_admission','ambiguous'),
                          ('confirmed_accuracy_eligible',True)):
            q,c,p=self.fixture()
            c[0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                self.build((q,c,p))

    def test_strict_and_leading_scores_are_separate_without_prompt_repair(self):
        q,c,p=self.fixture()
        p[0]['text']='A. explanation'
        summary,_,_=self.build((q,c,p))
        table=summary['tables']['original28_raw_labels']['M0']
        self.assertEqual((table['strict_correct'],table['content_correct'],table['format_compliant']),(27,28,27))

    def test_paired_corrections_and_losses_are_kept_even_when_net_zero(self):
        q,c,p=self.fixture()
        next(r for r in p if r['id']=='ovo:0' and r['arm']=='M2')['text']='B'
        next(r for r in p if r['id']=='ovo:1' and r['arm']=='LS_shared384_v1')['text']='B'
        summary,_,_=self.build((q,c,p))
        pair=summary['paired_comparisons']['historical_supported16']['M2']['LS_shared384_v1']['strict_correct']
        self.assertEqual(pair,dict(corrections=['ovo:0'],new_losses=['ovo:1'],net=0))

    def test_manual_declared_results_remain_separate_from_original_labels(self):
        q,c,p=self.fixture()
        for row in p:
            if row['id'].startswith('local_'):row['text']='B'
        summary,_,_=self.build((q,c,p))
        self.assertEqual(summary['tables']['original28_raw_labels']['M0']['strict_correct'],28)
        self.assertEqual(summary['tables']['manual6_source_declared']['M0']['strict_correct'],0)
        self.assertEqual(summary['human_review'],'pending')

    def test_review_queue_excludes_answers_truths_and_predictions(self):
        _,_,review=self.build(self.fixture())
        self.assertEqual(len(review),34)
        for row in review:
            self.assertFalse({'label_index','answer_index','text','predictions','strict_correct'} & set(row))
            self.assertEqual(row['review_status'],'pending')
            self.assertIsNone(row['semantic_answer_judgment'])

    def test_failed_answer_is_rejected_without_partial_score(self):
        q,c,p=self.fixture()
        p[0]['status']='failed'
        with self.assertRaises(ValueError):
            self.build((q,c,p))


class MockWorkerContracts(unittest.TestCase):
    def exercise(self,contract_failure=False):
        from experiments.wshy.long_video_gpu_2b import worker
        with tempfile.TemporaryDirectory() as directory:
            root=output_path(Path(directory)/'worker')
            (root/'cpu').mkdir(parents=True)
            (root/'cpu/preflight.json').write_text('{}',encoding='utf-8')
            (root/'gpu_protocol.json').write_text('{}',encoding='utf-8')
            calls=dict(loads=0,visual=0,hook=None,removed=False)
            tensor=SimpleNamespace(to=lambda device:None)
            builder=SimpleNamespace(pair=lambda *args,**kw:(SimpleNamespace(
                pixel_values_videos=tensor,video_grid_thw=tensor),None))
            def hook_register(callback):
                calls['hook']=callback
                return SimpleNamespace(remove=lambda:calls.update(removed=True))
            def get_features(*args,**kwargs):
                calls['visual']+=1
                calls['hook']()
                return SimpleNamespace(pooler_output=[None])
            model=SimpleNamespace(config=SimpleNamespace(text_config=SimpleNamespace(hidden_size=2048)),
                parameters=lambda:[SimpleNamespace(is_floating_point=lambda:True,dtype='fp32')],
                model=SimpleNamespace(visual=SimpleNamespace(register_forward_hook=hook_register),
                                      get_video_features=get_features))
            model.eval=lambda:model
            def load(*args,**kwargs):
                calls['loads']+=1
                return model
            cuda=SimpleNamespace(device_count=lambda:1,mem_get_info=lambda:(80*2**30,80*2**30),
                get_device_name=lambda:'mock GPU',max_memory_allocated=lambda:0,max_memory_reserved=lambda:0,
                memory_allocated=lambda:0,memory_reserved=lambda:0,synchronize=lambda:None,empty_cache=lambda:None)
            fake_torch=SimpleNamespace(cuda=cuda,float32='fp32',set_num_threads=lambda n:None,
                manual_seed=lambda n:None,inference_mode=nullcontext,cat=lambda *args:None)
            env=dict(torch=fake_torch,psutil=SimpleNamespace(virtual_memory=lambda:SimpleNamespace(available=80*2**30),
                Process=lambda:SimpleNamespace(memory_info=lambda:SimpleNamespace(rss=0))),
                Qwen3_5ForConditionalGeneration=SimpleNamespace(from_pretrained=load),
                AutoProcessor=SimpleNamespace(from_pretrained=lambda *a,**k:None),PairInputBuilder=lambda *a,**k:builder,
                fp32_reference=nullcontext,legacy=SimpleNamespace(features=SimpleNamespace(raw_block=lambda *a:None)),
                prompt=lambda q:q['question'])
            counts=[227,227,227,227,226,226]
            sources=[];queries=[];snapshots=[]
            for i,count in enumerate(counts):
                media=root/('video'+str(i)+'.mp4')
                media.write_bytes(b'mock media; never decoded')
                path='source'+str(i)
                frames=[dict(frame_id=j,pts_s=float(j),rgb_sha256='0'*64) for j in range(2*count+1)]
                sources.append(dict(source_path=path,media_path=media.name,media_sha256=sha(media),frames=frames))
                for j in range(6 if i<4 else 5):
                    serial=len(queries)
                    qid='ovo:'+str(serial) if serial<28 else 'local_longstream_v1:'+str(serial-27).zfill(2)
                    queries.append(dict(id=qid,source_path=path,query_time_s=float(2*count),
                        question='Which?',options=['first','second'],
                        cohort='official_OVO_unchanged' if serial<28 else 'manual_development'))
                    snapshots.append(dict(query_id=qid,received_frames=2*count+1,committed_pairs=count,
                        committed_through_s=float(2*count-1),observation_age_s=1.,pending_frame_id=2*count,
                        committed_frames_sha256='a'*64))
            def child(command,**kwargs):
                kwargs['stdout'].write('Ran 22 tests in 0.1s\n\n'+('FAILED\n' if contract_failure else 'OK\n'))
                return SimpleNamespace(returncode=int(contract_failure))
            def drive(frames,qs,samples,encode,readout,prompt,event,answer):
                count=len(frames)//2
                for i in range(count):
                    first=SimpleNamespace(frame_id=2*i,timestamp=float(2*i))
                    second=SimpleNamespace(frame_id=2*i+1,timestamp=float(2*i+1))
                    encode(i,first,second)
                    state=dict(valid=True,closed=False,visual_tokens=0,nodes=[],resident_bytes=0,
                        metadata_bytes=0,full_history_retained=False,language_cache_retained=False)
                    for arm in PROFILE['arms'][:-1]:
                        event(dict(event='append',arm=arm,state=state,pair=i))
                prefix=dict(arrived_frames=len(frames),committed_pairs=count,committed_through_s=float(2*count-1),
                    observation_age_s=1.,pending_frame=dict(frame_id=2*count),
                    historical_committed_frames_sha256='a'*64,query_padding=False)
                for q in qs:
                    for arm in PROFILE['arms']:
                        answer(dict(id=q['id'],arm=arm,source_path=q['source_path'],prefix=prefix,
                                    text='A',generated_ids=[32],status='ok'))
                    event(dict(event='query',id=q['id'],**prefix))
                return dict(sampled_frames=len(frames),encoded_pairs=count,reset_checks=5,queries=len(qs),answers=len(qs)*6)
            verified=(dict(PROFILE),dict(model='mock',resource_root=str(root)),dict(environment={'mock':True}),
                      (queries,sources,snapshots))
            with ExitStack() as stack:
                stack.enter_context(patch.dict('os.environ',dict(SLURM_JOB_ID='fixture',SLURM_CPUS_PER_TASK='4',
                    SLURM_RESTART_COUNT='0',CUDA_VISIBLE_DEVICES='0')))
                stack.enter_context(patch.dict('sys.modules',{'resource':SimpleNamespace(RUSAGE_SELF=0,
                    getrusage=lambda mode:SimpleNamespace(ru_maxrss=0))}))
                for name,value in (('verify_spec',lambda *a:verified),('verify_model',lambda *a:None),
                    ('runtime',lambda:env),('environment',lambda e:{'mock':True}),('run_sessions',drive),
                    ('decode_samples',lambda *a:iter(()))):
                    stack.enter_context(patch.object(worker,name,value))
                stack.enter_context(patch.object(worker.subprocess,'run',child))
                if contract_failure:
                    with self.assertRaisesRegex(RuntimeError,'before model load'):
                        worker.gpu(root,root/'gpu_protocol.json')
                    self.assertEqual(calls['loads'],0)
                    self.assertFalse((root/'gpu/load_started.json').exists())
                else:
                    worker.gpu(root,root/'gpu_protocol.json')
                    done=read(root/'gpu/completed.json')
                    self.assertEqual((calls['loads'],calls['visual']),(1,1360))
                    self.assertEqual((done['main_answers'],done['visual_forward_calls'],done['reset_checks']),(204,1360,30))
                    self.assertTrue(calls['removed'])
                    for name,expected in done['output_sha256'].items():
                        self.assertEqual(sha(root/'gpu'/name),expected)
                    with self.assertRaises(FileExistsError):
                        worker.gpu(root,root/'gpu_protocol.json')
                    self.assertEqual(calls['loads'],1)

    def test_mocked_worker_completes_fixed_grid_hashes_and_refuses_reload(self):
        self.exercise()

    def test_failed_new_contracts_stop_before_model_loading(self):
        self.exercise(contract_failure=True)


if __name__=='__main__':
    unittest.main()
