import contextlib
import io
import unittest
import torch
from units.model.val_official import ExpressionAnnotations
from units.model.tool.official_grec import RefExpEvaluator,generalized_box_iou


class OfficialGRECTests(unittest.TestCase):
    def evaluate(self,predictions):
        annotations=ExpressionAnnotations()
        annotations.add(1,[dict(bbox=[0,0,10,10],category_id=1)])
        annotations.add(2,[dict(bbox=[0,0,1,1],category_id=-1)])
        with contextlib.redirect_stdout(io.StringIO()):
            metric=RefExpEvaluator(annotations,('bbox',))
            metric.update(predictions)
            return metric.summarize()

    def prediction(self,boxes,scores):
        return dict(boxes=torch.tensor(boxes,dtype=torch.float64),scores=torch.tensor(scores,dtype=torch.float64))

    def test_threshold_boundary_and_empty_target(self):
        result=self.evaluate({1:self.prediction([[0,0,10,10]],[.7]),2:self.prediction([[0,0,1,1]],[.69])})
        self.assertEqual(result,dict(F1_score=1.,T_acc=1.,N_acc=1.))

    def test_duplicates_remain_and_fail_exact_set(self):
        result=self.evaluate({1:self.prediction([[0,0,10,10],[0,0,10,10]],[.9,.8]),2:self.prediction([[0,0,1,1]],[.8])})
        self.assertEqual(result['F1_score'],0.)
        self.assertEqual(result['N_acc'],0.)
        self.assertEqual(result['T_acc'],1.)

    def test_official_overlap_is_generalized_iou(self):
        boxes=torch.tensor([[0.,0.,10.,10.],[20.,20.,30.,30.]])
        overlap=generalized_box_iou(boxes,boxes[:1])
        self.assertEqual(overlap[0,0].item(),1.)
        self.assertLess(overlap[1,0].item(),0.)


if __name__=='__main__':unittest.main()
