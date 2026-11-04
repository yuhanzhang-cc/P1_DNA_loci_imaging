"""Independent-test interface scaffold. CLI is dry-run only in this phase.

Live inference is deliberately not enabled. Future cache evaluation requires
pre-threshold scores bound to both a validation-selected weight hash and an
independent dataset manifest hash. It never selects checkpoints/thresholds on test.
"""
from pathlib import Path
import argparse,json
from scaffold_common import read,digest,todos
from ranked_ap import ranked_point_ap


def verify_locks(locks,root):
    for lock in locks:
        checkpoint=(root/lock["checkpoint"]).resolve();selection=(root/lock["selection"]).resolve()
        if not checkpoint.is_relative_to(root.resolve()) or not selection.is_relative_to(root.resolve()):raise ValueError("Model lock escapes source root")
        if digest(checkpoint)!=lock["weight_sha256"] or digest(selection)!=lock["selection_sha256"]:raise ValueError("Frozen validation model lock differs")
        selected=read(selection)["primary"]
        if selected["checkpoint_sha256"]!=lock["weight_sha256"] or selected["calibrated_probability"]!=lock["threshold"]:
            raise ValueError("Checkpoint/threshold was not the validation-selected pair")
    return len(locks)


def evaluate_scored_cache(lock,cache,manifest_sha256):
    """Functional AP stage for a future inference adapter, tested synthetically."""
    if cache["weight_sha256"]!=lock["weight_sha256"] or cache["manifest_sha256"]!=manifest_sha256:
        raise ValueError("Candidate cache identity differs")
    if cache["dataset_role"]!="independent_test" or cache["independence_reviewed"] is not True:
        raise ValueError("No independent-test provenance confirmation")
    if cache["all_prethreshold_candidates"] is not True or cache["score_floor"] is not None:
        raise ValueError("Ranked AP requires complete unthresholded candidates")
    if not isinstance(cache.get("synthetic_fixture"),bool):raise ValueError("Candidate cache must explicitly identify synthetic versus real data")
    truth={sid:sample["truth"] for sid,sample in cache["images"].items()}
    candidates={sid:(sample["coordinates"],sample["scores"]) for sid,sample in cache["images"].items()}
    result=ranked_point_ap(truth,candidates,radius=3.)
    result.update(ranked_ap=result["ap"],locked_f1_threshold=lock["threshold"],test_threshold_selection=False,
                  engineering_only=cache["synthetic_fixture"],test_evaluated=not cache["synthetic_fixture"])
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--locks",type=Path,required=True);p.add_argument("--config",type=Path,required=True)
    p.add_argument("--dry-run",action="store_true",required=True)
    args=p.parse_args();spec=read(args.config);locks=read(args.locks)
    if spec["dataset_role"]!="independent_test" or spec["allow_existing_observed_internal_test"]:
        raise ValueError("Observed historical internal test cannot be called independent")
    count=verify_locks(locks,args.data_root.resolve())
    print(json.dumps(dict(passed=True,dry_run=True,validation_selected_locks_verified=count,
      pipeline=["verify_validation_selection","load_reviewed_independent_manifest_TODO","frozen_model_inference_adapter_TODO","full_scored_cache_binding","ranked_point_ap","export_new_test_readout"],
      launch_ready=False,unresolved=todos(spec),live_inference_adapter_implemented=False,
      independent_test_images_read=0,test_evaluated=False,optimizer_updates=0,threshold_selection_on_test=False),indent=2))


if __name__=="__main__":main()
