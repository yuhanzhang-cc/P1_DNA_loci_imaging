"""Collect initialization/backend evidence before CUDA smoke, never optimize."""
from pathlib import Path
import platform,subprocess,sys,traceback
from .common import write,fresh,ROOT,OLD,read


def diagnose(output,device="cpu"):
    output=fresh(output);result=dict(device=device,python=platform.python_version(),platform=platform.platform(),
        executable=sys.executable,optimizer_updates=0,cuda_available=False,passed=False)
    try:
        import numpy as np
        import scipy,torch
        from diagnostics.mechanism_v1.adapter import make_model
        from research_v4.engine import configure_device,reset_rng,model_hash
        result.update(torch=str(torch.__version__),numpy=np.__version__,scipy=scipy.__version__,
            cuda_runtime=torch.version.cuda,cudnn=torch.backends.cudnn.version(),cuda_available=torch.cuda.is_available(),
            torch_build=torch.__config__.show())
        result["cpu_capability"]=torch.backends.cpu.get_cpu_capability()
        result["host_processor"]=platform.processor()
        refs=ROOT/"configs/reference_initializations.json"
        if not refs.exists():refs=ROOT/OLD/"configs/reference_initializations.json"
        reference=read(refs)
        environment_path=ROOT/"configs/reference_environment.json"
        if not environment_path.exists():environment_path=ROOT/"docs/next_stage_plan_20261025/baseline_cuda_analysis_20261102/diagnostics/handoff_20261111/reference_environment.json"
        expected_environment=read(environment_path)
        result["reference_environment"]=expected_environment
        if device=="cuda" and not torch.cuda.is_available():raise RuntimeError("CUDA unavailable; no fallback")
        configure_device(device)
        if device=="cuda":
            result["gpu"]=torch.cuda.get_device_name(0)
            result["matched_reference_backend"]=result["gpu"]==expected_environment["gpu"] and str(torch.__version__)==expected_environment["torch"]
        evidence=[]
        for seed in (42,43,44):
            reset_rng(seed);h0=make_model("H0",device);h=model_hash(h0)
            reset_rng(seed);s0=make_model("S0",device);s=model_hash(s0)
            evidence.append(dict(seed=seed,h0_sha256=h,s0_sha256=s,paired_exact=h==s,
                historical_cuda_sha256=reference[str(seed)]["initial_model_state_sha256"],historical_exact=s==reference[str(seed)]["initial_model_state_sha256"]))
        result["initializations"]=evidence
        result["passed"]=all(r["paired_exact"] and (device!="cuda" or r["historical_exact"]) for r in evidence) and (device!="cuda" or result["matched_reference_backend"])
        if not result["passed"]:result["reason"]="Initialization mismatch: preserve evidence; do not waive or automatically retrain a baseline"
    except Exception as error:result.update(error=repr(error),traceback=traceback.format_exc())
    try:
        p=subprocess.run(["nvidia-smi"],capture_output=True,text=True,timeout=20)
        result["nvidia_smi"]=p.stdout+p.stderr
    except (OSError,subprocess.TimeoutExpired):result["nvidia_smi"]="unavailable"
    write(output/"doctor.json",result)
    return result
