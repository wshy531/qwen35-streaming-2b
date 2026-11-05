"""Tests for failures that would invalidate an empirical memory comparison."""
import copy
import unittest

from evaluation_gate import freeze_issues, official_identity, require_frozen, require_prediction_identity
from score_memory_eval import digest,score


def fixture():
    original=dict(id='toy:0',dataset='toy',source_id='clip',source_path='clip.mp4',
        task='TC',question='what happened first?',options=['left','right'],
        answer_index=0,query_time_s=10.,split='development')
    row=dict(original,source_group_id='recording0',parent_identity_status='unverified',
        media_sha256='1'*64,review=dict(verdict='supported_observation',
            reviewer='synthetic fixture, not a real review',note='toy witness',
            model_outputs_seen=False,flags_pending=[],witnesses=[dict(
                frame_id=0,pts_s=2.,source_path='clip.mp4',hash_verified=True,rgb_sha256='2'*64)]))
    registry=[dict(source_group_id='recording0',split='development')]
    return dict(status='frozen_for_evaluation',evaluation_scope='development_diagnostic',
        rows=[row],original_rows=[official_identity(original)],
        original_rows_sha256=digest([official_identity(original)]),
        split_registry=registry,split_registry_sha256=digest(registry),
        input_protocol=dict(model_revision='toy-fixed-revision',processor_revision='toy-fixed-revision',
            precision='float32',sampling_fps=.5,canvas_hw=[192,256],
            pairing_policy='two_real_or_snapshot_only_last_duplicate',
            query_cutoff_policy='pts_not_after_query',readout_layout='toy',
            prompt_template_sha256='3'*64,visual_token_budget=384,
            generation=dict(max_new_tokens=16,do_sample=False,enable_thinking=False),
            input_trace_sha256='4'*64,real_interface_receipt_sha256='5'*64))


class EvaluationGateTests(unittest.TestCase):
    def test_flag_alone_cannot_freeze_unreviewed_candidates(self):
        self.assertIn('original_question_snapshot_missing_or_duplicate',
                      freeze_issues(dict(status='frozen_for_evaluation',rows=[dict(id='x')])))
        require_frozen(fixture()) # synthetic schema acceptance, no real capability claim
        for change in ('pending','future','missing_hash','model_seen','label'):
            with self.subTest(change=change):
                b=fixture();r=b['rows'][0]
                if change=='pending':r['review']['verdict']='unresolved'
                elif change=='future':r['review']['witnesses'][0]['pts_s']=10.01
                elif change=='missing_hash':r['review']['witnesses'][0]['hash_verified']=False
                elif change=='model_seen':r['review']['model_outputs_seen']=True
                else:r['answer_index']=1
                with self.assertRaises(ValueError):require_frozen(b)

    def test_negative_label_requires_whole_prefix_review(self):
        b=fixture();b['rows'][0]['review']['verdict']='verified_negative'
        self.assertTrue(any('full_prefix' in i for i in freeze_issues(b)))
        b['rows'][0]['review'].update(full_prefix_reviewed=True,
                                     full_prefix_review_receipt_sha256='6'*64)
        require_frozen(b)

    def test_shared_wrong_prediction_identity_is_not_a_fair_pair(self):
        b=fixture();p=dict(evaluation_sha256=digest(b),input_trace_sha256='4'*64,
                           model_revision='toy-fixed-revision')
        require_prediction_identity(b,p)
        # Both arms could previously agree on the same wrong model/trace.
        for field in ('input_trace_sha256','model_revision'):
            wrong=dict(p);wrong[field]='wrong-but-same-in-both-arms'
            with self.assertRaises(ValueError):require_prediction_identity(b,wrong)

    def test_group_split_leakage_and_formal_parent_identity(self):
        b=fixture();b['split_registry'].append(dict(source_group_id='recording0',split='holdout'))
        b['split_registry_sha256']=digest(b['split_registry'])
        self.assertIn('source_recording_leaks_across_splits',freeze_issues(b))
        b=fixture();b['evaluation_scope']='holdout'
        self.assertTrue(any('parent_recording_not_verified' in i for i in freeze_issues(b)))

    def test_macro_groups_related_clips_by_parent(self):
        rows=[]
        for i,parent in enumerate(('a','a','b')):
            row=copy.deepcopy(fixture()['rows'][0]);row.update(id=str(i),source_id=str(i),source_group_id=parent)
            rows.append(row)
        s=score(rows,[dict(id='0',text='A')])['overall']
        self.assertEqual(s['videos'],2)
        self.assertEqual(s['micro_accuracy'],1/3)
        self.assertEqual(s['video_macro_accuracy'],.25)


if __name__=='__main__':
    unittest.main(verbosity=2)
