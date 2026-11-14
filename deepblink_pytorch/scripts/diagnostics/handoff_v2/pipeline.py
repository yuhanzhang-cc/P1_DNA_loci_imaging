"""Sequential CUDA plan, predeclared screening gate, automatic packaging.

No launch on import or dry-run. Scientific gate failure is a valid stop with
results, not an engineering failure or invitation to silently change the model.
"""
from pathlib import Path
import json,subprocess,sys,time,uuid
from .common import read,write,sha,ROOT
from .freeze import verify_frozen


def invocation(root,method,mode,seed=42,probe=None,resume=False):
    root=Path(root);decisions=root/"decisions"
    cmd=[sys.executable,"-u",str(root/"deepblink_pytorch/scripts/diagnostics/mechanism_v1/cli.py"),
        "--config",str(decisions/"frozen_configs"/(method+".json")),"--data-root",str(root),"--assets",str(root/"assets"),
        "--device","cuda","--seed",str(seed),"--"+mode]
    if mode=="smoke":cmd += ["--output",str(root/"outputs/probes"/(method+"_"+uuid.uuid4().hex[:12]))]
    else:
        cmd += ["--approval",str(decisions/(method+".json")),"--probe",str(probe)]
        if mode=="train":cmd += ["--output",str(root/"outputs/trials"/method/f"seed_{seed}")]
    if resume:cmd.append("--resume")
    return cmd


def plan(root):
    return dict(schema="paired_mechanism_pipeline_v2",dry_run=True,training_started=False,
        stages=["verify_sealed_kit","verify_real_frozen_decision","record_CUDA_environment_and_all_three_initializations",
            "S0_full_geometry_probe","P0_full_geometry_probe","S0_seed42","P0_seed42","compute_predeclared_development_gate",
            "if_pass_and_preapproved:S0_43,P0_43,S0_44,P0_44","verify_completed_artifacts","pack_results_with_checksums","local_ingest_and_report"],
        command_examples=[invocation(root,m,"train",42,"<automatically_registered_probe>") for m in ("S0","P0")],
        no_new_ordinary_baselines=True,test_arrays_allowed=False,maximum_formal_runs=6,
        true_remaining_inputs=["real image review","explicit research decision including P0 parameters","actual CUDA availability"])


def expansion_tasks(screen,decision):
    full=decision.get("completion_strategy","gate_conditioned")=="full_registered_six"
    return [(m,s) for s in (43,44) for m in ("S0","P0")] if (screen["passed"] or full) and decision["auto_expand_if_gate_passes"] else []


def execute(root,resume=False,run_command=None):
    root=Path(root).resolve()
    from diagnostics.mechanism_v1.controller import verify,check_complete
    from .doctor import diagnose
    from .archive import pack_return
    verify(root);marker,decision=verify_frozen(root/"decisions")
    if marker["synthetic_fixture"]:raise ValueError("Formal queue cannot use synthetic review")
    from filelock import FileLock
    out=root/"outputs/pipeline";out.mkdir(parents=True,exist_ok=True)
    state_path=out/"status.json"
    identity=dict(freeze_sha256=sha(root/"decisions/freeze_manifest.json"),bundle_sha256=sha(root/"bundle_manifest.json"))
    if state_path.exists():
        if not resume:raise ValueError("Existing pipeline state requires --resume")
        if read(state_path)["identity"]!=identity:raise ValueError("Pipeline decision/source changed")
    state=dict(status="preflight",identity=identity,completed_tasks=[],test_evaluated=False,engineering_only=False)
    launch=run_command or (lambda cmd:subprocess.run(cmd,cwd=root,check=True))
    def update(stage,**kwargs):
        state.update(status=stage,**kwargs,updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()));write(state_path,state,replace=True)
        print(json.dumps(dict(stage=stage,**kwargs)),flush=True)
    with FileLock(str(out/"queue.lock"),timeout=0):
        try:
            update("preflight")
            doctor=diagnose(out/("doctor_"+uuid.uuid4().hex[:12]),device="cuda")
            if not doctor["passed"]:raise RuntimeError("CUDA/backend/initialization gate failed; see doctor evidence")
            probes={}
            for method in ("S0","P0"):
                update("probe",method=method)
                cmd=invocation(root,method,"smoke");launch(cmd)
                path=Path(cmd[cmd.index("--output")+1])/"checks.json"
                if not read(path).get("passed"):raise ValueError("Probe failed")
                probes[method]=path
                launch(invocation(root,method,"ready-check",probe=path))
            update("probes_passed",probes={m:str(p) for m,p in probes.items()})
            def task(method,seed):
                folder=root/"outputs/trials"/method/f"seed_{seed}"
                if (folder/"status.json").exists() and read(folder/"status.json")["status"]=="complete":
                    check_complete(folder)
                    registration=read(folder/"registration.json")
                    if registration["binding"]["config_sha256"]!=marker["configurations"][method] or registration["approval_sha256"]!=__import__("diagnostics.mechanism_v1.contracts",fromlist=["canonical"]).canonical(read(root/"decisions"/(method+".json"))):
                        raise ValueError("Completed task does not belong to this frozen decision")
                else:
                    update("training",method=method,seed=seed)
                    launch(invocation(root,method,"train",seed,probes[method],resume=resume and (folder/"registration.json").exists()))
                    check_complete(folder)
                state["completed_tasks"].append(dict(method=method,seed=seed));update("task_completed",method=method,seed=seed)
            for method in ("S0","P0"):task(method,42)
            update("screening_gate")
            from diagnostics.mechanism_v1.runner import corpus
            from .gate import evaluate_gate
            valid=corpus(root,root/"assets").splits["valid"];truth={str(s):t for s,t in zip(valid["sample_id"],valid["y"])}
            gate=evaluate_gate(root/"outputs/trials/S0/seed_42",root/"outputs/trials/P0/seed_42",truth,decision["continuation_gate"])
            gate_path=out/("screening_gate_"+uuid.uuid4().hex[:12]+".json");write(gate_path,gate)
            update("screened",gate=gate)
            for method,seed in expansion_tasks(gate,decision):task(method,seed)
            update("packing")
            archive=pack_return(root,state["completed_tasks"],probes,gate_path)
            final="complete_six_runs" if len(state["completed_tasks"])==6 else "screen_complete_gate_not_passed" if not gate["passed"] else "screen_complete_expansion_not_authorized"
            update(final,return_archive=archive)
            return state
        except BaseException as error:
            import traceback
            update("failed_or_interrupted",error=repr(error),traceback=traceback.format_exc())
            raise


def status(root):
    root=Path(root);p=root/"outputs/pipeline/status.json"
    result=dict(pipeline=read(p) if p.exists() else dict(status="not_started"),tasks=[])
    for seed in (42,43,44):
        for method in ("S0","P0"):
            path=root/"outputs/trials"/method/f"seed_{seed}/status.json"
            result["tasks"].append(dict(method=method,seed=seed,**(read(path) if path.exists() else dict(status="not_started",epoch=0,updates=0))))
    return result
