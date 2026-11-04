"""Scaffold contracts and synthetic metrics only; no optimizer or baseline inference."""
from pathlib import Path
import copy
import json
import subprocess
import sys
import zipfile
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parent))
from scaffold_common import validate_config,matched_control,read,PARAMETERS
from ranked_ap import ranked_point_ap,step_ap
from weak_policy import classify,weak_recall,local_zscore
from formal_test_eval import evaluate_scored_cache
from aggregate_results import trial
from blind_audit import synthetic_example,validate_responses,check_public_zip,agreement


def config():
    return dict(id="P0",status="not_started",training_enabled=False,adapter_implemented=False,expected_parameters=PARAMETERS,seeds=[42,43,44],
      architecture=dict(base="H0_frozen_structure",input_channels=1,head_channels=3,added_trainable_parameters=0),
      components=dict(signed_offsets=True,neighborhood_supervision=True),supervision=dict(radius_px=None),
      recipe=dict(epochs=100,updates=2200,batch_size=8,canvas=512,optimizer="Adam",lr=.001,period=5,threshold_count=31,precision="float32"))


def policy():
    return dict(policy_id="legacy_train_q25_local_zscore_v1",threshold=2.4566830772038957,comparison="strict_less_than",radius_px=10,std_ddof=0,std_epsilon=1e-6)


def test_training_cli_refuses_non_dry_run(tmp_path):
    path=tmp_path/"config.json";path.write_text(json.dumps(config()))
    result=subprocess.run([sys.executable,str(Path(__file__).with_name("train_mechanism.py")),"--config",str(path)],capture_output=True,text=True)
    assert result.returncode!=0 and "--dry-run" in result.stderr


def test_capacity_match_does_not_invent_a_measured_adapter():
    cfg=config();control=matched_control(cfg)
    assert cfg["components"]["neighborhood_supervision"] and not control["components"]["neighborhood_supervision"]
    assert control["architecture"]==cfg["architecture"] and control["expected_parameters"]==671719
    assert control["alias_of"]=="S0" and not control["counts_as_new_trial"]
    assert not validate_config(cfg)["launch_ready"]


@pytest.mark.parametrize("change",["capacity","recipe","illegal","enabled"])
def test_changed_or_illegal_training_contract_rejected(change):
    cfg=config()
    if change=="capacity":cfg["architecture"]["added_trainable_parameters"]=1
    elif change=="recipe":cfg["recipe"]["epochs"]=1
    elif change=="illegal":cfg["components"]["signed_offsets"]=False
    else:cfg["training_enabled"]=True
    with pytest.raises(ValueError):validate_config(cfg)


def test_ranked_ap_matches_known_sklearn_example():
    import sklearn
    from sklearn.metrics import average_precision_score
    truth={"synthetic":np.asarray([[0,0],[10,10]])}
    candidates={"synthetic":(np.asarray([[50,50],[30,30],[10,10],[0,0]]),np.asarray([.1,.4,.35,.8]))}
    result=ranked_point_ap(truth,candidates,1.)
    expected=average_precision_score([0,0,1,1],[.1,.4,.35,.8])
    assert result["ap"]==pytest.approx(expected)==pytest.approx(5/6)
    print(f"synthetic only: sklearn {sklearn.__version__} reference PASS; AP={expected}")


@pytest.mark.parametrize("seed",range(6))
def test_ranked_ap_all_scores_ties_and_missing_truth_match_sklearn(seed):
    from sklearn.metrics import average_precision_score
    rng=np.random.default_rng(seed)
    labels=rng.integers(0,2,40);scores=rng.choice([.1,.4,.7,.9],40)
    ntrue=int(labels.sum())+5
    expected=average_precision_score(labels,scores)*labels.sum()/ntrue
    assert step_ap(labels,scores,ntrue)["ap"]==pytest.approx(expected)


def test_ranked_ap_does_not_rematch_like_frozen_hungarian_f1():
    truth={"a":np.asarray([[0,0],[0,2]])}
    candidates={"a":(np.asarray([[0,1],[0,0]]),np.asarray([.9,.8]))}
    result=ranked_point_ap(truth,candidates,1.1)
    assert [r["is_tp"] for r in result["ranked_records"]]==[1,0]
    assert result["ap"]==.5


def test_ranked_ap_empty_and_duplicate_detections():
    empty=np.empty((0,2))
    assert ranked_point_ap({"a":empty},{"a":(empty,np.asarray([]))})["ap"] is None
    assert ranked_point_ap({"a":np.asarray([[0,0]])},{"a":(empty,np.asarray([]))})["ap"]==0
    result=ranked_point_ap({"a":np.asarray([[0,0]])},{"a":(np.asarray([[0,0],[0,0]]),np.asarray([.5,.5]))})
    assert result["ap"]==.5 and sum(r["is_tp"] for r in result["ranked_records"])==1


def test_ranked_ap_rejects_nonfinite_scores_and_image_mismatch():
    with pytest.raises(ValueError):step_ap([1],[float("nan")],1)
    with pytest.raises(ValueError):ranked_point_ap({"a":[]},{"b":([],[])})
    with pytest.raises(ValueError):step_ap([.5],[.9],1)
    with pytest.raises(ValueError):step_ap([1],[.9],1.5)


def test_weak_cutoff_is_strict_and_missing_is_not_strong():
    cutoff=policy()["threshold"]
    values=[np.nextafter(cutoff,-np.inf),cutoff,np.nextafter(cutoff,np.inf),np.nan]
    assert classify(values,policy()).tolist()==["weak","non_weak","non_weak","unclassifiable"]
    assert weak_recall(values,[0,1],policy())==dict(weak_n=1,weak_tp=1,weak_recall=1.,unclassifiable_n=1)
    assert weak_recall([np.nan,9],[1],policy())["weak_recall"] is None


def test_weak_original_pixel_feature_and_constant_image():
    image=np.zeros((7,7),np.float32);image[3,3]=100
    assert local_zscore(image,3,3,policy())==pytest.approx(np.sqrt(48),rel=1e-5)
    assert np.isnan(local_zscore(np.zeros((7,7)),3,3,policy()))
    wrong=policy();wrong["threshold"]+=.1
    with pytest.raises(ValueError):classify([1],wrong)
    with pytest.raises(ValueError):weak_recall([1],[0,0],policy())


def test_future_not_started_results_are_blank_not_zero(tmp_path):
    row=trial(tmp_path,"P0",42)[0]
    assert row["status"]=="not_started" and row["f1"] is None


def test_formal_test_cache_requires_full_scores_and_identity():
    lock=dict(weight_sha256="synthetic-weight",threshold=.5)
    cache=dict(weight_sha256="synthetic-weight",manifest_sha256="synthetic-manifest",dataset_role="independent_test",independence_reviewed=True,
               all_prethreshold_candidates=True,score_floor=None,synthetic_fixture=True,images={"synthetic":dict(truth=[[0,0]],coordinates=[[0,0]],scores=[.9])})
    result=evaluate_scored_cache(lock,cache,"synthetic-manifest")
    assert result["ap"]==1. and not result["test_threshold_selection"]
    assert result["ranked_ap"]==1. and result["engineering_only"] and not result["test_evaluated"]
    wrong=copy.deepcopy(cache);wrong["score_floor"]=.5
    with pytest.raises(ValueError):evaluate_scored_cache(lock,wrong,"synthetic-manifest")
    wrong=copy.deepcopy(cache);wrong["weight_sha256"]="foreign"
    with pytest.raises(ValueError):evaluate_scored_cache(lock,wrong,"synthetic-manifest")


def test_blind_validator_rejects_incomplete_duplicate_and_leaked_rows():
    key,data=synthetic_example(None)
    assert validate_responses(data[0],key,complete=True,synthetic=True)["responses_complete"]
    wrong=copy.deepcopy(data[0]);wrong[0]["reviewer"]=""
    with pytest.raises(ValueError):validate_responses(wrong,key,complete=True,synthetic=True)
    wrong=copy.deepcopy(data[0]);wrong[-1]=wrong[0]
    with pytest.raises(ValueError):validate_responses(wrong,key,synthetic=True)
    wrong=copy.deepcopy(data[0]);wrong[0]["method"]="H0"
    with pytest.raises(ValueError):validate_responses(wrong,key,synthetic=True)
    with pytest.raises(ValueError):validate_responses(data[0],key,synthetic=False)


def test_blind_presence_is_not_detection_tp_and_key_cannot_leak(tmp_path):
    key,data=synthetic_example(None)
    wrong=copy.deepcopy(data[0]);wrong[0]["adjudication_phase"]="blind"
    with pytest.raises(ValueError):validate_responses(wrong,key,synthetic=True)
    result,cross=agreement(data,key)
    assert len(result)==6 and len(cross)>0
    with pytest.raises(ValueError):agreement([data[0],data[0]],key)
    path=tmp_path/"leaked.zip"
    with zipfile.ZipFile(path,"w") as z:z.writestr("private/source_key.csv","secret")
    with pytest.raises(ValueError):check_public_zip(path)
