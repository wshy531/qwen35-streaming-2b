"""Fixed-budget LS policy, original controls and six-arm integration contracts."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from experiments.wshy.online_memory_2b.frozen import ARMS, policies
from experiments.wshy.online_memory_2b.session import MemorySession, digest
from experiments.wshy.online_memory_2b.shared import EventNode, SharedHistory384


def block(i, width=32):
    return policies().features.raw_block(torch.randn(48,width,
        generator=torch.Generator().manual_seed(8100+i)), i, 2.*i, 2.*i+1)


def append(session, i, width=32):
    return session.append_pair(block(i,width), frame_ids=(2*i,2*i+1))


class SharedContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_first_eight_pairs_remain_complete_even_in_one_event(self):
        bank = SharedHistory384()
        for i in range(8):
            bank.append(block(i), starts_event=i==0)
            self.assertEqual(len(bank.nodes()), i+1)
            self.assertTrue(all(n.metadata()['exact_pair'] for n in bank.nodes()))
            self.assertEqual(bank.within_merges+bank.long_merges, 0)
        self.assertEqual(bank.status()['tokens'], 384)
        old = policies().forest384.Forest384()
        for i in range(8): old.append(block(i),starts_event=i==0)
        self.assertEqual(old.status()['tokens'], 240)
        self.assertEqual(old.within_merges, 3)

    def test_ninth_pair_only_merges_history_and_keeps_recent_four_exact(self):
        bank = SharedHistory384()
        for i in range(9): bank.append(block(i), starts_event=i==0)
        self.assertEqual(bank.within_merges, 1)
        self.assertEqual(bank.long_merges, 0)
        self.assertEqual(len(bank.history), 4)
        self.assertEqual([e.node.first for e in bank.recent], [5,6,7,8])
        self.assertTrue(all(e.node.metadata()['exact_pair'] for e in bank.recent))
        self.assertEqual(bank.status()['history_tokens'],192)

    def test_same_frozen_penalty_and_explicit_lowest_index_tie(self):
        bank = SharedHistory384()
        for i in range(8): bank.append(block(i), starts_event=True)
        with patch.object(bank.legacy,'penalties',return_value=[{'total':2.},{'total':1.},{'total':1.},{'total':3.}]) as costs:
            bank.append(block(8),starts_event=True)
        self.assertEqual(costs.call_count,1)
        self.assertEqual([(e.node.first,e.node.last) for e in bank.history],[(0,0),(1,2),(3,3),(4,4)])
        self.assertEqual(bank.long_merges,1)

    def test_frozen_numeric_merge_matches_selected_pair(self):
        bank = SharedHistory384()
        for i in range(8): bank.append(block(i),starts_event=i==0)
        expected=bank.legacy.merge(bank.history[0].node,bank.history[1].node,within=True)
        with patch.object(bank.legacy,'penalties',return_value=[{'total':float(i)} for i in range(4)]):
            bank.append(block(8),starts_event=False)
        self.assertEqual(digest([expected]),digest([bank.history[0].node]))

    def test_event_boundary_follows_its_pair_through_delayed_promotion(self):
        bank = SharedHistory384()
        for i in range(8): bank.append(block(i),starts_event=i in (0,3,6))
        self.assertEqual([(e.first_event,e.last_event) for e in bank.history],[(0,0),(0,0),(0,0),(1,1)])
        self.assertEqual([e.first_event for e in bank.recent],[1,1,2,2])

    def test_mixed_event_nodes_do_not_use_within_event_merge(self):
        bank = SharedHistory384()
        for i in range(8): bank.append(block(i),starts_event=True)
        with patch.object(bank.legacy,'penalties',return_value=[{'total':0.}]*4):
            bank.append(block(8),starts_event=True)
            bank.append(block(9),starts_event=True)
        self.assertEqual(bank.long_merges,2)
        self.assertEqual(bank.within_merges,0)
        self.assertEqual((bank.history[0].first_event,bank.history[0].last_event),(0,2))

    def test_512_pairs_preserve_mass_ranges_recent_and_storage_bounds(self):
        bank=SharedHistory384()
        fields=set(bank.__dict__)
        for i in range(512):
            bank.append(block(i),starts_event=i%7==0)
            state,nodes=bank.status(),bank.nodes()
            self.assertEqual(set(bank.__dict__),fields)
            self.assertLessEqual(state['tokens'],384)
            self.assertLessEqual(state['history_tokens'],192)
            self.assertLessEqual(state['metadata_bytes'],32768)
            self.assertLessEqual(state['resident_tensor_bytes'],8*7488)
            self.assertLessEqual(len(bank.history),4)
            self.assertLessEqual(len(bank.recent),4)
            self.assertEqual(nodes[0].first,0)
            self.assertEqual(nodes[-1].last,i)
            self.assertTrue(all(a.last+1==b.first for a,b in zip(nodes,nodes[1:])))
            self.assertAlmostEqual(sum(float(n.mass.sum()) for n in nodes),48*(i+1),delta=1e-3)
            self.assertEqual(bank.within_merges+bank.long_merges,max(0,i+1-8))
            for e in bank.recent:
                self.assertEqual(digest([e.node]),digest([bank.legacy.raw(block(e.node.first))]))

    def test_slow_independent_list_reference_matches_every_prefix(self):
        legacy=policies().forest
        for period in (1,9,1000):
            bank=SharedHistory384()
            recent,history=[],[]
            event=-1
            for i in range(64):
                starts=i%period==0
                event=0 if i==0 else event+int(starts)
                entry=(legacy.raw(block(i)),event,event)
                recent.append(entry)
                if len(recent)>4:history.append(recent.pop(0))
                while len(history)>4:
                    costs=legacy.penalties([e[0] for e in history],history[-1][0].end)
                    index=min(range(len(costs)),key=lambda j:(costs[j]['total'],j))
                    a,b=history[index:index+2]
                    within=len({a[1],a[2],b[1],b[2]})==1
                    history[index:index+2]=[(legacy.merge(a[0],b[0],within=within),a[1],b[2])]
                bank.append(block(i),starts_event=starts)
                self.assertEqual(digest([e[0] for e in history+recent]),digest(bank.export()))
                self.assertEqual([(e.first_event,e.last_event) for e in bank.history],[(e[1],e[2]) for e in history])

    def test_descriptor_and_boundary_decisions_match_M2_and_A06(self):
        sessions=[MemorySession(a) for a in ('M2','M2_A06','LS_shared384_v1')]
        for i in range(40):
            records=[append(s,i) for s in sessions]
            self.assertTrue(all(r['starts_event']==records[0]['starts_event'] for r in records))
            self.assertTrue(all(r['adjacent_distance']==records[0]['adjacent_distance'] for r in records))
            self.assertTrue(all(torch.equal(s.previous,sessions[0].previous) for s in sessions))

    def test_2048_width_resident_budget_includes_event_descriptor(self):
        session=MemorySession('LS_shared384_v1')
        for i in range(16):append(session,i,2048)
        self.assertEqual(session.status()['detector_bytes'],8192)
        self.assertLessEqual(session.status()['resident_bytes'],384*(2048*4+4+24)+8192)

    def test_ask_and_no_ask_have_identical_continuation(self):
        asked=MemorySession('LS_shared384_v1',lambda ns,q:dict(text='A',generated_ids=[1]))
        control=MemorySession('LS_shared384_v1')
        for i in range(48):
            append(asked,i);append(control,i)
            before=torch.get_rng_state().clone()
            self.assertTrue(asked.ask('q')['memory_unchanged'])
            self.assertEqual(asked.state_digest(),control.state_digest())
            self.assertTrue(torch.equal(before,torch.get_rng_state()))

    def test_exported_nodes_do_not_alias_and_mutating_reader_is_rejected(self):
        session=MemorySession('LS_shared384_v1')
        for i in range(12):append(session,i)
        before=session.state_digest()
        session.export()[0].features.fill_(99)
        self.assertEqual(session.state_digest(),before)
        def mutate(ns,q):
            ns[0].mass.zero_()
            return dict(text='A')
        session.readout=mutate
        with self.assertRaisesRegex(RuntimeError,'exported input'):session.ask('q')
        self.assertEqual(session.state_digest(),before)

    def test_event_range_counter_and_descriptor_mutation_are_detected(self):
        for field in ('range','counter','descriptor'):
            session=MemorySession('LS_shared384_v1')
            for i in range(10):append(session,i)
            def mutate(ns,q):
                if field=='range':
                    session.bank.history[0]=replace(session.bank.history[0],last_event=999)
                elif field=='counter':session.bank.current_event+=1
                else:session.previous.add_(1)
                return dict(text='A')
            session.readout=mutate
            with self.assertRaisesRegex(RuntimeError,'persistent memory'):session.ask('q')
            self.assertFalse(session.valid)

    def test_reader_exception_keeps_bank_and_allows_continuation(self):
        session=MemorySession('LS_shared384_v1')
        append(session,0)
        before=session.state_digest()
        def fail(ns,q):raise ValueError('reader failure')
        session.readout=fail
        with self.assertRaisesRegex(ValueError,'reader failure'):session.ask('q')
        self.assertEqual(session.state_digest(),before)
        append(session,1)

    def test_invalid_pairs_and_direct_flags_are_atomic(self):
        session=MemorySession('LS_shared384_v1')
        append(session,0)
        before=session.state_digest()
        invalid=[(block(0),(2,3)),(block(2),(2,3)),(block(1,64),(2,3)),(block(1),(1,2)),(block(1),(2,2))]
        bad=block(1);bad.features[0,0]=float('nan');invalid.append((bad,(2,3)))
        for value,frames in invalid:
            with self.assertRaises(ValueError):session.append_pair(value,frame_ids=frames)
            self.assertEqual(session.state_digest(),before)
        with self.assertRaises(ValueError):session.bank.append(block(1),starts_event='yes')
        self.assertEqual(session.state_digest(),before)

    def test_odd_tail_is_terminal_and_reset_releases_event_metadata(self):
        session=MemorySession('LS_shared384_v1')
        for i in range(16):append(session,i)
        value=session.legacy.features.raw_block(torch.zeros(48,32),16,32.,32.)
        session.append_pair(value,frame_ids=(32,32),padded_tail=True)
        with self.assertRaises(ValueError):append(session,17)
        session.reset()
        self.assertEqual(session.state_digest(),MemorySession('LS_shared384_v1').state_digest())
        self.assertEqual(session.bank.current_event,-1)
        self.assertEqual(session.status()['resident_bytes'],0)
        append(session,0);session.close()
        with self.assertRaises(RuntimeError):append(session,0)
        with self.assertRaises(RuntimeError):session.ask('q')
        self.assertEqual(session.status()['resident_bytes'],0)

    def test_original_default_methods_remain_unchanged(self):
        self.assertEqual(ARMS,('M0','M2','reservoir4','text_only'))
        from experiments.wshy.online_memory_2b.protocol import FIXED_PROTOCOL
        self.assertEqual(FIXED_PROTOCOL['main_answers'],684)

    def test_six_arm_protocol_keeps_171_questions_and_explicit_extra_budget(self):
        from experiments.wshy.online_memory_2b.protocol import LONG_SHORT_PROTOCOL, LONG_SHORT_ARMS
        self.assertEqual(LONG_SHORT_ARMS,('M0','M2','M2_A06','LS_shared384_v1','reservoir4','text_only'))
        self.assertEqual(LONG_SHORT_PROTOCOL['main_answers'],171*6)
        self.assertEqual(LONG_SHORT_PROTOCOL['maximum_answers'],171*6+8*3*5)
        self.assertEqual(LONG_SHORT_PROTOCOL['maximum_generated_tokens'],(171*6+8*3*5)*8)

    def test_changed_arm_budget_or_unknown_schema_cannot_use_fixed_protocol(self):
        from experiments.wshy.online_memory_2b.protocol import LONG_SHORT_PROTOCOL, verify_run
        for field,value in (('arms',['M0']),('maximum_answers',9999),('maximum_generated_tokens',99999),('schema','unknown')):
            with tempfile.TemporaryDirectory() as directory:
                p=Path(directory)
                (p/'protocol.json').write_text(json.dumps(dict(LONG_SHORT_PROTOCOL,**{field:value})),encoding='utf-8')
                with self.assertRaisesRegex(ValueError,'protocol'):
                    verify_run(p)

    def test_cross_arm_mutation_during_question_invalidates_all_banks(self):
        from experiments.wshy.online_memory_2b.runner import guarded_banks
        sessions={a:MemorySession(a) for a in ('M2','M2_A06','LS_shared384_v1')}
        for s in sessions.values():append(s,0)
        def mutate():
            sessions['M2'].previous.zero_()
            return dict(text='A')
        with self.assertRaisesRegex(RuntimeError,'comparison banks'):
            guarded_banks(sessions,mutate)
        self.assertTrue(all(not s.valid for s in sessions.values()))

    def test_six_arm_readout_guard_accepts_independent_text_and_visual_answers(self):
        from experiments.wshy.online_memory_2b.protocol import LONG_SHORT_ARMS
        from experiments.wshy.online_memory_2b.runner import guarded_banks
        sessions={a:MemorySession(a,lambda ns,q:dict(text='A')) for a in LONG_SHORT_ARMS if a!='text_only'}
        for s in sessions.values():append(s,0)
        for arm in LONG_SHORT_ARMS:
            result=guarded_banks(sessions,lambda:sessions[arm].ask('q') if arm in sessions else dict(text='A'))
            self.assertEqual(result['text'],'A')
        self.assertTrue(all(s.valid for s in sessions.values()))

    def test_new_candidates_are_not_required_to_match_old_answers(self):
        from experiments.wshy.online_memory_2b.runner import historical_replay
        for arm in ('M2_A06','LS_shared384_v1'):
            self.assertIsNone(historical_replay({},'q',arm,dict(text='different'),'frames'))
        with self.assertRaises(ValueError):historical_replay({},'q','unknown',{},'frames')

    def test_missing_or_changed_historical_baseline_still_fails(self):
        from experiments.wshy.online_memory_2b.runner import historical_replay
        result=dict(generated_ids=[1],input_ids_sha256='input',node_metadata=[])
        ref=dict(result,input_frames_sha256='frames')
        expected={('q','M2'):ref}
        self.assertTrue(all(historical_replay(expected,'q','M2',result,'frames').values()))
        changed=dict(result,generated_ids=[2])
        self.assertFalse(all(historical_replay(expected,'q','M2',changed,'frames').values()))
        with self.assertRaises(KeyError):historical_replay({},'q','M2',result,'frames')

    def scoring_fixture(self, run):
        # Candidate rows intentionally copy frozen M2 solely to test the data
        # path. This fixture is not a new candidate capability measurement.
        from experiments.wshy.online_memory_2b.frozen import PACKAGE
        from experiments.wshy.online_memory_2b.protocol import (read,rows,sha,save,
            LONG_SHORT_ARMS,LONG_SHORT_PROTOCOL,checkpoint_grid,validate_requests)
        requests=read(PACKAGE/'bundle/runs/2b-short-event-inputs-20261020-01/requests.json')
        references={(r['id'],r['arm']):r for r in rows(PACKAGE/'evidence/short-event-v1/result/main/predictions.jsonl') if r['arm'] in ARMS}
        protocol=dict(LONG_SHORT_PROTOCOL,checkpoints=checkpoint_grid(validate_requests(requests)))
        save(run/'protocol.json',protocol)
        save(run/'scoring_ONLY.json',read(PACKAGE/'catalog/evaluated171_catalog.json'))
        out=run/'gpu';out.mkdir()
        with (out/'predictions.jsonl').open('x',encoding='utf-8') as f:
            for request in requests:
                for arm in LONG_SHORT_ARMS:
                    ref=references[request['id'],arm if arm in ARMS else 'M2']
                    p=dict(ref,arm=arm,status='ok',replay={'historical':True} if arm in ARMS else None)
                    f.write(json.dumps(p)+'\n')
        with (out/'probes.jsonl').open('x',encoding='utf-8') as f:
            for h,points in protocol['checkpoints'].items():
                for n in points:
                    for arm in LONG_SHORT_ARMS:
                        if arm!='text_only':
                            f.write(json.dumps(dict(media_sha256=h,pairs=n,arm=arm,
                                memory_unchanged=True,prefix_only=True,accuracy_scored=False))+'\n')
        (out/'events.jsonl').write_text('',encoding='utf-8')
        save(out/'completed.json',dict(status='complete',protocol_sha256=sha(run/'protocol.json'),
            output_sha256={n:sha(out/n) for n in ('predictions.jsonl','probes.jsonl','events.jsonl')}))
        return protocol,requests

    def test_offline_six_arm_scoring_compares_with_M2_and_A06_without_claiming_new_accuracy(self):
        from experiments.wshy.online_memory_2b.runner import score
        from experiments.wshy.online_memory_2b.protocol import read
        with tempfile.TemporaryDirectory() as directory:
            run=Path(directory)
            protocol,requests=self.scoring_fixture(run)
            with patch('experiments.wshy.online_memory_2b.runner.verify_run',return_value=(protocol,requests)):
                score(run)
            summary=read(run/'score/summary.json')
            self.assertEqual(summary['main_answers'],1026)
            self.assertEqual(len(read(run/'score/details.json')),1026)
            self.assertEqual(summary['paired_comparisons']['M2']['LS_shared384_v1']['strict_correct']['net'],0)
            self.assertEqual(summary['paired_comparisons']['M2_A06']['LS_shared384_v1']['strict_correct']['net'],0)
            self.assertEqual(summary['paired_against_M0']['M2']['strict_correct']['net'],1)
            review=read(run/'score/review_without_predictions.json')
            self.assertEqual(len(review),171)
            self.assertTrue(all(r['independent_review_status']=='pending' for r in review))
            self.assertFalse(any('prediction' in r or 'answer_index' in r for r in review))

    def test_changed_comparison_outputs_are_rejected_before_scoring(self):
        from experiments.wshy.online_memory_2b.runner import score
        with tempfile.TemporaryDirectory() as directory:
            run=Path(directory)
            protocol,requests=self.scoring_fixture(run)
            with (run/'gpu/predictions.jsonl').open('a',encoding='utf-8') as f:f.write('\n')
            with patch('experiments.wshy.online_memory_2b.runner.verify_run',return_value=(protocol,requests)):
                with self.assertRaisesRegex(ValueError,'outputs changed'):score(run)


if __name__=='__main__':unittest.main()
