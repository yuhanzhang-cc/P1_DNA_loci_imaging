"""Read-only readiness checks and synthetic P0 target coverage, no optimization."""
from pathlib import Path
import argparse,json
import numpy as np
from .contracts import ROOT,REL,read,validate,write_new,canonical
from .adapter import encode,make_model
from .runner import corpus
from research_v4.engine import reset_rng,model_hash,preserve_rng
from deepblink_torch.round2_protocol import training_orders


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--data-root",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();data=corpus(a.data_root)
    config=read(ROOT/REL/"configs/S0.json");p0=read(ROOT/REL/"configs/P0.json")
    validate(config);validate(p0,require_parameters=False)
    refs=read(ROOT/REL/"configs/reference_initializations.json");rows=[]
    for seed in [42,43,44]:
        with preserve_rng():
            reset_rng(seed);h0=make_model("H0");h= model_hash(h0)
            reset_rng(seed);s0=make_model("S0");s=model_hash(s0)
            reset_rng(seed);p=make_model("P0");ph=model_hash(p)
        if h!=s or s!=ph:raise AssertionError("Paired initialization differs")
        sampler=canonical(training_orders(176,100,seed))
        if sampler!=refs[str(seed)]["sampler_orders_sha256"]:raise AssertionError("Sampler changed")
        rows.append(dict(seed=seed,local_h0_s0_p0_initialization_exact=True,initial_state_sha256=s,
            historical_cuda_initialization_equal=s==refs[str(seed)]["initial_model_state_sha256"],sampler_exact=True))
    # Fixture only: never write these illustrative P0 choices into its real config.
    fixture=dict(p0);fixture["supervision"]=dict(radius_px=2,ownership="nearest_truth_then_index",normalization="per_truth_mean")
    coverage=[]
    for split,d in data.splits.items():
        points=0;mass_error=0.
        for truth,shape in zip(d["y"],d["original_shape"]):
            target,owner=encode(truth,shape,512,fixture)
            for i in range(len(truth)):mass_error=max(mass_error,abs(float(target[...,3][owner==i].sum())-1.))
            points+=len(truth)
        coverage.append(dict(split=split,images=len(d["x"]),truth_points=points,max_per_truth_weight_mass_error=mass_error))
    protected=read(a.data_root/"docs/next_stage_plan_20261025/baseline_cuda_analysis_20261102/diagnostics/scaffold_20261103/evidence/immutable_before.json")
    from .contracts import sha
    bad=[name for name,digest in protected.items() if sha(Path(name))!=digest]
    if bad:raise AssertionError("Historical protected files changed: "+str(bad[:3]))
    result=dict(passed=True,optimizer_updates=0,gpu_used=False,test_evaluated=False,protected_files_unchanged=len(protected),
        paired_initialization=rows,p0_fixture_only=True,p0_real_parameters_still_todo=p0["supervision"],p0_target_coverage=coverage,
        formal_training_count=41,formal_launch_ready=False,remaining_gates=["human mechanism review","P0 blind audit and parameters","actual CUDA source-bound probe and historical initialization check"])
    write_new(a.output,result);print(json.dumps(result))
if __name__=="__main__":main()
