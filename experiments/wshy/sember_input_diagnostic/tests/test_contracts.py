import copy
import unittest
from PIL import Image
from experiments.wshy.sember_input_diagnostic.controls import rotate,matched_dense,canvas,question_prompt,ARMS
from experiments.wshy.sember_input_diagnostic.scoring import build


class Controls(unittest.TestCase):
    def query(self):
        return dict(id='q',question='Which?',options=['A. A. Smith','B. blue','C. cat','D. dog','E. egg'],
                    source_path='video.mp4',query_time_s=9.)
    def f(self,i): return dict(frame_id=i,pts_s=i/2,rgb_sha256=str(i))
    def test_rotation_identity_and_content(self):
        q=self.query(); saved=copy.deepcopy(q)
        self.assertEqual(question_prompt(rotate(q,0)[0]),question_prompt(q))
        for s in range(5):
            r,order=rotate(q,s)
            for j,i in enumerate(order): self.assertEqual(r['options'][j][3:],q['options'][i][3:])
        self.assertEqual(q,saved)
    def test_all_positions_and_inverse(self):
        for truth in range(5):
            positions=[]
            for s in range(5):
                _,order=rotate(self.query(),s); d=(truth-s)%5
                self.assertEqual(order[d],truth); positions.append(d)
            self.assertEqual(sorted(positions),list(range(5)))
    def test_bad_shift(self):
        for s in (-1,5,True):
            with self.assertRaises(ValueError): rotate(self.query(),s)
    def test_mixed_labels(self):
        q=self.query();q['options'][1]='blue'
        with self.assertRaises(ValueError): rotate(q,1)
    def test_dense_keeps_anchors_and_bounds(self):
        a=[self.f(i) for i in (0,2,4,6)]
        rows,omit=matched_dense(a,[self.f(i) for i in range(9)])
        self.assertEqual([r['frame_id'] for r in rows],[0,1,2,3,4,6])
        self.assertEqual(omit,self.f(5))
    def test_short_no_fake_frames(self):
        a=[self.f(0),self.f(2)]
        rows,omit=matched_dense(a,[self.f(1)])
        self.assertEqual(rows,a);self.assertEqual(omit,self.f(1))
    def test_even_union_no_omission(self):
        a=[self.f(i) for i in (0,2,4,6)]
        rows,omit=matched_dense(a,[self.f(1),self.f(3)])
        self.assertEqual(len(rows),6); self.assertIsNone(omit)
    def test_conflicting_identity(self):
        with self.assertRaises(ValueError): matched_dense([self.f(0),self.f(2)],[dict(self.f(2),pts_s=1.1)])
    def test_odd_anchors_rejected(self):
        with self.assertRaises(ValueError): matched_dense([self.f(0)],[])
    def test_canvas_preserves_aspect_and_alignment(self):
        class Frame:
            def to_image(self): return Image.new('RGB',(720,960),(255,0,0))
        low=canvas(Frame(),192,256); high=canvas(Frame(),288,384)
        self.assertEqual(low.size,(256,192)); self.assertEqual(high.size,(384,288))
        self.assertEqual(low.getpixel((0,0)),(0,0,0)); self.assertEqual(high.getpixel((0,0)),(0,0,0))
        self.assertEqual(high.getpixel((192,144)),(255,0,0))
    def predictions(self, unknown=False):
        q=self.query(); result=[]
        for a in ARMS:
            for s in range(5):
                result.append(dict(id='q',arm=a,shift=s,source_path=q['source_path'],query_time_s=9.,
                    status='ok',display_to_original=rotate(q,s)[1],text='unsure' if unknown else 'ABCDE'[(4-s)%5]+' because evidence'))
        return result
    def labels(self):
        return [dict(id='q',correct_index=4,correct_letter='E',question_category='detail',answer_start_time=0.,answer_end_time=2.)]
    def test_rotated_scoring_uses_content(self):
        summary,detail=build([self.query()],self.labels(),self.predictions())
        self.assertTrue(all(r['correct'] for r in detail))
        self.assertTrue(all(v['content_correct']==5 for v in summary['totals'].values()))
        self.assertTrue(all(v['content_consistency']['all_five_same']==1 for v in summary['totals'].values()))
        self.assertNotIn('format',str(summary))
    def test_unparsed_remains_pending(self):
        summary,detail=build([self.query()],self.labels(),self.predictions(True))
        self.assertTrue(all(r['correct'] is None for r in detail))
        self.assertIsNone(summary['totals']['B']['mean_content_accuracy'])
        self.assertEqual(summary['paired']['R_vs_B']['evaluable_pairs'],0)
    def test_duplicate_and_bad_mapping_rejected(self):
        p=self.predictions()
        with self.assertRaises(ValueError): build([self.query()],self.labels(),p+[p[0]])
        p[0]['display_to_original']=[4,3,2,1,0]
        with self.assertRaises(ValueError): build([self.query()],self.labels(),p)


if __name__=='__main__': unittest.main()
