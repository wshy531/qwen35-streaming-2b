import copy
import unittest
from PIL import Image
from experiments.wshy.sember_resolution_2b.controls import (
    PRESETS,canvas,geometry,replay_ids,schedule,validate_history,expected_keys)
from experiments.wshy.sember_resolution_2b.scoring import build
from experiments.wshy.sember_input_diagnostic.controls import canvas as old_canvas


def fixture():
    queries=[dict(id=f'q{i:03}',source_path=f'v{i%18:02}',query_time_s=100.,
        question='Which?',options=['A. apple','B. book','C. cat','D. dog','E. egg']) for i in range(102)]
    labels=[dict(id=q['id'],correct_index=0,correct_letter='A',question_category='detail',
        answer_start_time=0.,answer_end_time=4.) for q in queries]
    old=[]
    for q in queries:
        for arm in ('B','R','F','T'):
            for s in range(5):
                old.append(dict(id=q['id'],source_path=q['source_path'],query_time_s=q['query_time_s'],
                    arm=arm,shift=s,status='ok',text='ABCDE'[(-s)%5]+' because it is shown.',
                    display_to_original=[(j+s)%5 for j in range(5)],generated_ids=[10+s],
                    input_ids_sha256='i',position_ids_sha256='p',frames_sha256='f',pixels_sha256='x'))
    reused=validate_history(queries,old); fresh=[]
    panel=set(replay_ids(queries))
    for r in reused:
        if r['id'] in panel: fresh.append(dict(r,provenance='replay',answer_seconds=1.,input_tokens=100))
    for q in queries:
        for arm,res in [('native','high')]+[(a,r) for a in ('LS','A06') for r in PRESETS]:
            for s in range(5):
                fresh.append(dict(id=q['id'],source_path=q['source_path'],query_time_s=q['query_time_s'],
                    arm=arm,resolution=res,shift=s,status='ok',text='ABCDE'[(-s)%5],
                    display_to_original=[(j+s)%5 for j in range(5)],generated_ids=[10+s],
                    input_ids_sha256='i',position_ids_sha256='p',frames_sha256='f',pixels_sha256='x',
                    provenance='new',answer_seconds=1.,input_tokens=100,encode_seconds=2.,
                    bank_update_seconds=.1,bank_peak_bytes=1000))
    return queries,labels,old,reused,fresh


class Controls(unittest.TestCase):
    def test_geometry_and_budget(self):
        self.assertEqual([geometry(r)['pair_tokens'] for r in PRESETS],[48,108,192])
        self.assertEqual([geometry(r)['total_tokens'] for r in PRESETS],[384,864,1536])
    def test_geometry_unknown(self):
        with self.assertRaises(KeyError): geometry('default')
    def test_endpoint_pixel_identity(self):
        for size in ((177,311),(640,480),(481,161),(1,1)):
            im=Image.linear_gradient('L').resize(size).convert('RGB')
            class Frame:
                def to_image(self): return im.copy()
            for r,hw in (('low',(192,256)),('mid',(288,384))):
                self.assertEqual(canvas(Frame(),r).tobytes(),old_canvas(Frame(),*hw).tobytes())
    def test_all_presets_preserve_aspect(self):
        class Frame:
            def to_image(self): return Image.new('RGB',(400,300),'white')
        for r,(h,w) in PRESETS.items():
            im=canvas(Frame(),r)
            self.assertEqual(im.size,(w,h)); self.assertEqual(im.getbbox(),(0,0,w,h))
    def test_canvas_does_not_mutate_original(self):
        im=Image.new('RGB',(640,480),'red')
        class Frame:
            def to_image(self): return im.copy()
        for r in PRESETS: canvas(Frame(),r)
        self.assertEqual(im.size,(640,480))
    def test_schedule_counts(self):
        q,*_=fixture(); s=schedule(q)
        self.assertEqual((s['maximum_new_answers'],s['maximum_new_tokens'],s['unique_analysis_answers']),
                         (3840,30720,5100))
        self.assertEqual(s['new_native']+s['new_memory']+s['replay_answers'],3840)
    def test_schedule_rejects_missing(self):
        q,*_=fixture()
        with self.assertRaises(ValueError): schedule(q[:-1])
    def test_schedule_rejects_duplicate(self):
        q,*_=fixture(); q[-1]=q[0]
        with self.assertRaises(ValueError): schedule(q)
    def test_panel_order_invariant(self):
        q,*_=fixture(); self.assertEqual(replay_ids(q),replay_ids(list(reversed(q))))
        self.assertEqual(len(replay_ids(q)),18)
    def test_history_reuse_counts_and_immutability(self):
        q,_,old,_,_=fixture(); saved=copy.deepcopy(old); rows=validate_history(q,old)
        self.assertEqual(len(rows),1530); self.assertEqual(old,saved)
        self.assertFalse(any(r.get('historical_arm')=='F' for r in rows))
        self.assertEqual({r['resolution'] for r in rows if r['historical_arm']=='R'},{'mid'})
    def test_history_mapping_rejected(self):
        q,_,old,_,_=fixture(); old[0]['display_to_original']=[1,0,2,3,4]
        with self.assertRaises(ValueError): validate_history(q,old)
    def test_history_duplicate_rejected(self):
        q,_,old,_,_=fixture()
        with self.assertRaises(ValueError): validate_history(q,old+[old[0]])
    def test_score_no_replay_double_count(self):
        q,l,_,r,n=fixture(); summary,details=build(q,l,r,n)
        self.assertEqual(len(details),5100); self.assertEqual(len(summary['totals']),10)
        self.assertTrue(all(v['content_correct']==510 for v in summary['totals'].values()))
        self.assertEqual(summary['replay_excluded_from_accuracy'],270)
        self.assertTrue(all(v['answers']==90 for v in summary['matched_costs'].values()))
        self.assertNotIn('strict_correct',str(summary))
    def test_score_missing_new_answer_rejected(self):
        q,l,_,r,n=fixture()
        with self.assertRaises(ValueError): build(q,l,r,n[:-1])
    def test_score_replay_drift_rejected(self):
        q,l,_,r,n=fixture(); n[0]['generated_ids']=[999]
        with self.assertRaises(ValueError): build(q,l,r,n)
    def test_primary_grid_text_not_tripled(self):
        q,*_=fixture(); keys=expected_keys(q)
        self.assertEqual(sum(a=='text_only' for _,a,_,_ in keys),510)
    def test_unresolved_not_silently_scored_correct(self):
        q,l,_,r,n=fixture()
        row=next(x for x in n if x['arm']=='LS'); row['text']='uncertain'
        summary,_=build(q,l,r,n)
        self.assertEqual(summary['totals']['LS/low']['unresolved'],1)
        self.assertIsNone(summary['totals']['LS/low']['mean_content_accuracy'])


if __name__=='__main__': unittest.main()
