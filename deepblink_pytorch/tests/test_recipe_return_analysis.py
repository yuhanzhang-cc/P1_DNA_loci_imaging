"""Independent small examples for the returned-trial analysis estimators."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from analyze_recipe_return import aggregate,exact_paired,holm


class RecipeReturnTests(unittest.TestCase):
    def test_exact_test_has_correct_three_pair_floor(self):
        self.assertEqual(exact_paired([1,1,1]),.25)
        self.assertEqual(exact_paired([.009,-.0002,-.0083]),1.)
        self.assertEqual(exact_paired([0,0,0]),1.)

    def test_holm_keeps_original_order(self):
        result=holm([.02,.01,.4])
        for a,b in zip(result,[.04,.03,.4]):self.assertAlmostEqual(a,b)

    def test_micro_is_not_macro_and_localization_is_match_weighted(self):
        images=[dict(tp=1,fp=0,fn=0,weak_tp=1,weak_n=1,bright_tp=0,bright_n=0,f1=1.,n_pred=1,n_true=1,distance_sum=.2,distance_sq_sum=.04),
                dict(tp=2,fp=2,fn=2,weak_tp=0,weak_n=1,bright_tp=2,bright_n=3,f1=.5,n_pred=4,n_true=4,distance_sum=1.,distance_sq_sum=.5)]
        m=aggregate(images)
        self.assertAlmostEqual(m["f1"],.6);self.assertAlmostEqual(m["macro_f1"],.75)
        self.assertAlmostEqual(m["weak_recall"],.5);self.assertAlmostEqual(m["bright_recall"],2/3)
        self.assertAlmostEqual(m["mean_error_px"],.4)


if __name__=="__main__":unittest.main()
