"""Whole-study task catalog and runnable existing-model adapters, without GPU work."""
from pathlib import Path
import numpy as np
import torch
import yaml
from .common import dump,write_csv,read_json,rows
from .engine import MODELS,make_model,forward,reset_rng
from .augmentation import prepare_sample
from .queues import queues
from deepblink_torch.losses import combined_dice_rmse_loss
from deepblink_torch.round2_protocol import dense_heatmap_loss


EXTERNAL_FIELDS=["sample_id","image_file","label_file","height","width","coordinate_unit","um_per_pixel",
                 "biological_group_id","acquisition_id","source_container","collection_date","dataset_role",
                 "image_sha256","label_sha256","annotation_version","independence_notes"]


def run(ctx):
    ctx.task("study_scaffold","running")
    out=ctx.out/"06_study_scaffold";out.mkdir(exist_ok=True)
    registry=[];rng=np.random.default_rng(612)
    image=rng.normal(size=(128,128)).astype(np.float32);truth=np.asarray([[21.2,39.3],[70.4,92.5]],np.float32)
    for key in MODELS:
        reset_rng(42);model=make_model(key,"cpu")
        x,y,_,_=prepare_sample(image,truth,canvas=128,dense=key=="H0")
        prediction=forward(model,torch.from_numpy(x[None]),torch.from_numpy(y[None]))
        loss=(dense_heatmap_loss if key=="H0" else combined_dice_rmse_loss)(torch.from_numpy(y[None]),prediction)
        loss.backward()
        finite=bool(torch.isfinite(prediction).all() and torch.isfinite(loss))
        if not finite:raise AssertionError("Existing-model adapter failed: "+key)
        registry.append(dict(key=key,model_name=MODELS[key],parameters=sum(p.numel() for p in model.parameters()),
                              output_shape=list(prediction.shape),finite_loss=float(loss.detach()),forward_backward_passed=True,
                              optimizer_steps=0,representation="dense_heatmap_offset" if key=="H0" else "grid4"))
        del model,loss,prediction
    dump(out/"model_adapter_checks.json",dict(passed=True,models=registry,new_training_tasks=0))
    dump(out/"gpu_queues.json",queues())
    pending=[]
    for queue,items in queues().items():
        for item in items:pending.append(dict(queue=queue,**item,status="deferred_cuda_not_run",actual_new_epochs=0))
    write_csv(out/"gpu_task_manifest.csv",pending)
    # Preserve historical results as historical; no reruns to fill a cosmetic matrix.
    existing=[]
    for study in ("next_stage_v3","next_stage_v3_detail_completion","next_stage_v3_closing_controls"):
        for p in sorted((ctx.data_root/"outputs"/study/"01_screen").glob("*/seed_*/selection.json")):
            j=read_json(p);choice=j["primary"]
            existing.append(dict(study=study,method=p.parent.parent.name,seed=int(p.parent.name.split("_")[-1]),
                                 status="completed_archived",f1=choice["f1"],evidence="development_validation",path=p.relative_to(ctx.data_root).as_posix()))
    cloud=ctx.data_root/"outputs/next_stage_spotiflow_20261030/06_cuda_results/extracted/outputs/spotiflow_cuda"
    for seed in (42,43,44):
        p=cloud/f"seed_{seed}/selection.json";j=read_json(p)
        existing.append(dict(study="Spotiflow_CUDA_finetune",method="Spotiflow_F",seed=seed,status="completed_archived",
                             f1=j["primary"]["f1"],evidence="development_validation_pretrained_transfer",path=p.relative_to(ctx.data_root).as_posix()))
    for row in rows(ctx.data_root/"docs/next_stage_plan_20261025/experiment_registry.csv"):
        if row["study_id"] in ("NET_A","NET_B","CORE_D","CORE_D2"):
            if not (ctx.data_root/row["artifact_or_planned_location"]).is_dir():raise ValueError("Historical v2 artifact missing")
            existing.append(dict(study="next_stage_v2",method=row["study_id"],seed=42,status="completed_archived",
                                 f1=float(row["metric_value"]),evidence="development_v2_different_selection_rule",path=row["artifact_or_planned_location"]))
    write_csv(out/"existing_trial_index.csv",existing)
    matrix=dict(status="not_executable_core_method_not_selected",new_training_tasks_started=0,
        requires=["review-supported bottleneck","substantive related-work difference","frozen core implementation",
                  "same training recipe and supervision","capacity control","valid mechanism-removal definition"],
        prior_completed_controls=["R1 vs S1","R2 vs S2","R1 vs G0","R1 vs Qglobal"],
        cells=[dict(id=i,role=role,seeds=[42,43,44],model_key=None,enabled=False) for i,role in
               (("base","original_backbone_with_final_recipe"),("capacity","simple_capacity_matched_alternative"),
                ("full","final_core_method"),("minus","legal_trainable_mechanism_removal"))],
        prospective_training_cap=12,automatic_training_allowed=False,
        reason="No supported final core architecture yet; do not rename old controls as a novel method")
    (out/"core_ablation_template.yaml").write_text(yaml.safe_dump(matrix,sort_keys=False),encoding="utf-8")
    from .ablation import validate
    dump(out/"core_ablation_gate.json",validate(matrix))
    write_csv(out/"external_data_collection_template.csv",[],EXTERNAL_FIELDS)
    instructions="""# External data handoff

Keep new images and annotations outside all frozen training archives. Fill the CSV template with relative image/label paths, pixel y,x labels, original geometry, acquisition/container and biological group IDs, collection date, annotation version and SHA256 values. Use dataset_role=external_new. Unknown group identity must remain unknown; do not fabricate biological replicates. Current own-model inference supports original FOVs no larger than512 pixels; larger-image tiling needs a separate locked policy.

Keep raw acquisition files and calibration metadata. Blank um_per_pixel is allowed for pixel-only evaluation; never replace it with a guessed1.0. Label CSV headers must be y,x. All frozen models and thresholds must be declared before evaluation; no per-image tuning or new threshold selection on external data.

The external evaluator validates hashes, dimensions, units, group metadata and overlap against development pixels/source metadata. Passing metadata checks is not a proof of biological independence. A scientific reviewer must confirm the acquisition design. The template contains no sample data and no acquisition has been performed by this task.
"""
    (out/"EXTERNAL_DATA_README.md").write_text(instructions,encoding="utf-8")
    roadmap=[dict(stage="data_and_protocol",engineering="implemented",scientific_gate="annotation_and_independence_pending"),
             dict(stage="training_recipe",engineering="runner_and_six_task_queue",scientific_gate="deferred_GPU_training"),
             dict(stage="baseline",engineering="classical_and_existing_model_adapters",scientific_gate="R0_H0_scratch_queues_deferred_GPU"),
             dict(stage="core_ablation",engineering="prior_result_index_and_disabled_matrix_validator_contract",scientific_gate="core_method_not_selected"),
             dict(stage="generalization",engineering="manifest_and_frozen_threshold_evaluator",scientific_gate="new_data_collection_pending"),
             dict(stage="reporting",engineering="unified_metrics_source_locks_and_reports",scientific_gate="final_research_claims_not_established")]
    write_csv(out/"study_coverage.csv",roadmap)
    result=dict(passed=True,existing_model_adapters=len(registry),existing_trial_records=len(existing),
        recipe_gpu_tasks=6,optional_baseline_gpu_tasks=9,core_ablation_tasks_enabled=0,new_data_collected=False,
        new_training_tasks_started=0,scope_extension="2026-11-02 user authorized continued whole-study scaffolding; GPU/data collection packaged and deferred")
    dump(out/"checks.json",result)
    ctx.task("study_scaffold","completed",**result)
    return result
