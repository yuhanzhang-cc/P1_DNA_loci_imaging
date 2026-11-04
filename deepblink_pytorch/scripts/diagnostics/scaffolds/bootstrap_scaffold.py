"""Create additive templates once; no existing study or review files overwritten."""
from pathlib import Path
import argparse
import copy
from scaffold_common import base,save_new,csv_new,matched_control,digest,read


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--data-root",type=Path,required=True)
    args=p.parse_args();root=args.data_root.resolve();out=base(root)
    template=dict(id="S0",status="not_started",training_enabled=False,adapter_implemented=False,counts_as_new_trial=True,
       note="scaffold only, not run",review_decision="TODO: 待 blind_audit 回填后定并确认机制",expected_parameters=671719,
       architecture=dict(base="H0_frozen_structure",input_channels=1,head_channels=3,added_trainable_parameters=0),
       components=dict(signed_offsets=True,neighborhood_supervision=False),
       supervision=dict(radius_px=0,ownership="not_applicable_single_anchor",normalization="anchor_mean"),
       recipe=dict(epochs=100,updates=2200,batch_size=8,canvas=512,optimizer="Adam",lr=.001,period=5,threshold_count=31,precision="float32"),
       implementation_hash="TODO: no architecture/target implementation in this phase",seeds=[42,43,44])
    full=copy.deepcopy(template);full.update(id="P0")
    full["components"]["neighborhood_supervision"]=True
    full["supervision"]=dict(radius_px=None,radius_candidate_px=2,ownership=None,normalization=None,
          decision="TODO: 待 blind_audit 回填后定；候选nearest-truth ownership and per-truth averaging")
    for name,config in (("S0",template),("P0",full),("CAP_P0",matched_control(full))):save_new(out/f"configs/{name}.json",config)
    registry=dict(status="scaffold only, not run",formal_completed_training_count=41,new_training_started=0,tasks=[])
    for method in ("S0","P0","CAP_P0"):
        for seed in (42,43,44):registry["tasks"].append(dict(method=method,seed=seed,status="not_started",epoch=0,updates=0,
             alias_of="S0" if method=="CAP_P0" else None,counts_as_new_trial=method!="CAP_P0"))
    save_new(out/"configs/registry.json",registry)
    matrix=dict(status="scaffold only, not run",automatic_training_allowed=False,primary_contrast="P0-S0",
      components=["signed_offsets","neighborhood_supervision"],cells=[
        dict(id="base",config=None,existing_method="H0_cuda_reference",signed_offsets=False,neighborhood_supervision=False,reuse_existing=True),
        dict(id="capacity",config="S0.json",signed_offsets=True,neighborhood_supervision=False,reuse_existing=False),
        dict(id="full",config="P0.json",signed_offsets=True,neighborhood_supervision=True,reuse_existing=False),
        dict(id="minus_support",config="CAP_P0.json",signed_offsets=True,neighborhood_supervision=False,alias_of="S0",reuse_existing=False),
        dict(id="minus_signed",config=None,signed_offsets=False,neighborhood_supervision=True,expected_invalid=True,
             reason="bounded [0,1] offsets cannot represent neighborhood targets; do not train an illegal control")],
      independent_new_methods=["S0","P0"],prospective_new_runs=6,not_twelve=True)
    save_new(out/"configs/ablation_matrix.json",matrix)
    weak=dict(policy_id="legacy_train_q25_local_zscore_v1",feature="local_zscore",threshold=2.4566830772038957,
       comparison="strict_less_than",radius_px=10,std_ddof=0,std_epsilon=1e-6,center_rounding="python_round_ties_to_even",
       input="original unpadded float32 image",invalid="unclassifiable_not_nonweak",zero_weak_recall=None,
       cutoff_source="frozen training-only q25; never refit on validation/test",strata=["weak","non_weak","unclassifiable"],
       diagnostic_quartile_edges=[2.402340366424743,3.4319434818684265,4.7150023567686326],quartile_edges_source="existing 372-point development diagnostic only; not the weak threshold")
    save_new(out/"configs/weak_policy.json",weak)
    test=dict(stage="scaffold only, not run",dataset_role="independent_test",dataset_manifest=None,
       independence_reviewed=False,exclusions_json=None,candidate_cache=None,live_inference_adapter_enabled=False,
       decision="TODO: 独立数据采集/来源审查与正式test协议确认后定",allow_existing_observed_internal_test=False,
       matching_radius_px=3.,ap_policy="ranked_point_ap_greedy_step_v1",test_evaluated=False)
    save_new(out/"configs/independent_test.json",test)
    base_results=root/"outputs/next_stage_v4_recipe_20261101"
    locations=dict(R0=base_results/"extracted_r0/outputs/research_v4_cuda/baseline_r0/R0_cuda_reference",
                   H0=base_results/"extracted_h0/outputs/research_v4_cuda/baseline_h0/H0_cuda_reference",
                   G0=base_results/"08_returned_gpu/extracted/outputs/research_v4_cuda/recipe/G0_ref_cuda")
    locks=[]
    for method,folder in locations.items():
        for seed in (42,43,44):
            trial=folder/f"seed_{seed}";selection=read(trial/"selection.json")["primary"]
            locks.append(dict(method=method,seed=seed,checkpoint=str((trial/"best_model.pt").relative_to(root)).replace("\\","/"),
                weight_sha256=digest(trial/"best_model.pt"),selection=str((trial/"selection.json").relative_to(root)).replace("\\","/"),
                selection_sha256=digest(trial/"selection.json"),threshold=selection["calibrated_probability"],validation_selected=True))
    save_new(out/"configs/validation_model_locks.json",locks)
    public=root/"docs/next_stage_plan_20261025/baseline_cuda_analysis_20261102/diagnostics/overnight_ready/blind_audit_public"
    fields=["anonymous_id","raw_file","marked_file","candidate_is_real_spot","number_of_spots","background_description",
            "suggested_center_y_patch_px","suggested_center_x_patch_px","review_confidence","reviewer","notes",
            "adjudicated_detection_label","adjudication_phase","synthetic_fixture"]
    from scaffold_common import table
    template_rows=[dict(r,adjudicated_detection_label="",adjudication_phase="",synthetic_fixture="False") for r in table(public/"blind_audit_manifest.csv")]
    csv_new(out/"templates/blind_audit_response_template.csv",template_rows,fields)
    save_new(out/"templates/blind_audit_column_schema.json",dict(expected_cases=60,required_completed=["candidate_is_real_spot","number_of_spots","review_confidence","reviewer"],
       presence_values=["yes","no","uncertain"],confidence_values=["low","medium","high"],
       detection_values=["TP","FP","uncertain"],detection_requires_phase="post_blind_with_annotations",
       biological_spot_presence_is_not_one_to_one_detection_tp=True,source_mapping_must_remain_private=True))
    print("BOOTSTRAP PASS: additive templates; 6 future unique runs + 3 aliases; completed count stays 41")


if __name__=="__main__":main()
