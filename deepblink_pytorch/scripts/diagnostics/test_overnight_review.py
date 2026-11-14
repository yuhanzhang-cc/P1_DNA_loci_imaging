"""Spatial set arithmetic, no invented labels, deterministic anonymous sampling."""
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
from prepare_overnight_review import spatial_fp_sets, geometric_tag, crop_patch, sample_cases, source_range_check
from research_v4.common import write_csv
from deepblink_torch.evaluation import match_coordinates


def test_net_fp_difference_is_not_the_extra_candidate_set():
    h=dict(coords=np.asarray([[1,1],[5,5],[9,9]],np.float32),fp_indices=np.asarray([0,1,2]))
    g=dict(coords=np.asarray([[1.1,1],[30,30]],np.float32),fp_indices=np.asarray([0,1]))
    common,honly,gonly=spatial_fp_sets(h,g,.5)
    assert common=={0:0}
    assert honly==[1,2] and gonly==[1]
    assert len(honly)-len(gonly)==1 and len(honly)==2


def test_geometry_does_not_assert_background_truth():
    truth=np.asarray([[10,10],[10,14]],np.float32)
    prediction=np.asarray([[10,10],[10,14],[10,11]],np.float32)
    match=match_coordinates(prediction,truth,3.)
    assert geometric_tag(np.asarray([10,9]),truth,match)["geometry_class"]=="nearby_duplicate_proxy"
    assert geometric_tag(np.asarray([10,12]),truth,match)["geometry_class"]=="assignment_conflict_proxy"
    assert geometric_tag(np.asarray([30,30]),truth,match)["geometry_class"]=="other_requires_visual_review"


def test_crop_preserves_native_pixels_and_marks_unobserved_padding():
    image=np.arange(80*80,dtype=np.float32).reshape(80,80)
    patch,bounds=crop_patch(image,0.,0.)
    assert patch.shape==(64,64) and bounds==(-32,-32,32,32)
    assert np.isnan(patch[:32]).all()
    np.testing.assert_array_equal(patch[32:,32:],image[:32,:32])


def test_sampling_is_reproducible_and_reports_small_pools():
    pools={k:[dict(stratum=k,sample_id=f"synthetic_{k}_{i}",seed=42,true_index=i,y=40.,x=40.,method="",pred_index=None,event_id="")
                for i in range(n)] for k,n in (("common_tp",30),("h0_only_tp",4),("h0_only_fp",30))}
    a,counts=sample_cases(pools,20)
    b,_=sample_cases(pools,20)
    assert a==b and counts["h0_only_tp"]["selected"]==4
    assert len(a)==44


def test_source_box_bound_and_unreachable_truth_are_reported(tmp_path):
    table=[dict(method="H0",cpu_source_linked="True",sample_id="synthetic",seed=42,true_index=0,
                true_y=1.5,true_x=.5,peak_anchor_y=0,peak_anchor_x=0,dy_px=-.6,dx_px=0)]
    path=tmp_path/"peak.csv";write_csv(path,table)
    result=source_range_check(path,tmp_path)
    assert result["unreachable_target_events"]==1 and result["minimum_bound_mean_px"]==.5


def test_source_box_bound_cannot_exceed_observed_residual(tmp_path):
    import pytest
    table=[dict(method="H0",cpu_source_linked="True",sample_id="synthetic",seed=42,true_index=0,
                true_y=1.5,true_x=.5,peak_anchor_y=0,peak_anchor_x=0,dy_px=-.1,dx_px=0)]
    path=tmp_path/"peak.csv";write_csv(path,table)
    with pytest.raises(ValueError,match="representability bound"):
        source_range_check(path,tmp_path)
