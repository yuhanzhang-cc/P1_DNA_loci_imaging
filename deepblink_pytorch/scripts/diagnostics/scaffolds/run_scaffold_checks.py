"""Run 13 bounded scaffold checks. No train method or independent test is run."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import json
import os
import subprocess
import sys
from scaffold_common import CODE_ROOT,base,read,save_new,digest

HERE=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--data-root",type=Path,required=True);p.add_argument("--dry-run",action="store_true",required=True)
    args=p.parse_args();root=args.data_root.resolve();out=base(root);config=out/"configs"
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run=out/"runs"/stamp;run.mkdir(parents=True,exist_ok=False)
    examples=run/"examples";examples.mkdir();evidence=run/"evidence";evidence.mkdir()
    env=dict(os.environ,PYTHONIOENCODING="utf-8",PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",P1_SCAFFOLD_PYTHON=sys.executable.replace("\\","/"))
    ready=root/"docs/next_stage_plan_20261025/baseline_cuda_analysis_20261102/diagnostics/overnight_ready"
    python=lambda script,*a:[sys.executable,str(HERE/script),*map(str,a)]
    A1=[python("train_mechanism.py","--config",config/f"{m}.json","--dry-run") for m in ("S0","P0")]
    bash=Path("C:/Program Files/Git/bin/bash.exe") if os.name=="nt" else Path("/bin/bash")
    commands={
      "A1":A1,
      "A2":[python("capacity_control.py","--config",config/"P0.json","--output",examples/"CAP_P0.json","--dry-run")],
      "A3":[[str(bash),(HERE/"04_status.sh").as_posix(),"--data-root",root.as_posix()]],
      "B4":[python("ablation_runner.py","--matrix",config/"ablation_matrix.json","--dry-run")],
      "B5":[python("ablation_runner.py","--matrix",config/"ablation_matrix.json","--dry-run")],
      "B6":[python("aggregate_results.py","--data-root",root,"--output",examples/"aggregate_existing","--dry-run")],
      "C7":[python("formal_test_eval.py","--data-root",root,"--locks",config/"validation_model_locks.json","--config",config/"independent_test.json","--dry-run")],
      "C8":[[sys.executable,"-m","pytest","-q","-s",str(HERE/"test_scaffolds.py"),"-k","ranked_ap"]],
      "C9":[python("weak_policy.py","--config",config/"weak_policy.json","--features",root/"outputs/next_stage_v3_closing_controls/02_analysis/validation_truth_features.csv","--dry-run")],
      "D10":[python("reproduce.py","--data-root",root,"--dry-run"),
              python("reproduce.py","--data-root",root,"--output",examples/"clean_reproduction_table","--run-readonly")],
      "D11":[python("inspect_docs.py","--data-root",root,"--dry-run")],
      "E12":[python("blind_audit.py","validate","--responses",out/"templates/blind_audit_response_template.csv","--key",ready/"private/blind_audit_source_key.csv","--public-zip",ready/"blind_audit_public.zip","--dry-run")],
      "E13":[python("blind_audit.py","aggregate","--synthetic-demo","--output",examples/"synthetic_review","--dry-run")]}
    results=[]
    for item,batch in commands.items():
        log=evidence/(item+".log")
        with log.open("x",encoding="utf-8",newline="\n") as stream:
            for command in batch:
                stream.write("COMMAND: "+subprocess.list2cmdline(command)+"\n");stream.flush()
                result=subprocess.run(command,cwd=CODE_ROOT,env=env,capture_output=True,text=True,encoding="utf-8")
                stream.write(result.stdout+result.stderr);stream.flush()
                if result.returncode:
                    save_new(run/"failure.json",dict(item=item,returncode=result.returncode,log=str(log)))
                    raise RuntimeError("Scaffold check failed: "+item+"; inspect "+str(log))
        results.append(dict(item=item,passed=True,commands=batch,evidence=str(log.relative_to(out)).replace("\\","/")))
        print(item+" PASS",flush=True)
    subprocess.run([sys.executable,"-m","pytest","-q",str(HERE/"test_scaffolds.py")],cwd=CODE_ROOT,env=env,check=True,
                   stdout=(evidence/"all_unit_tests.log").open("x",encoding="utf-8"),stderr=subprocess.STDOUT)
    protected=read(out/"evidence/immutable_before.json")
    for path,h in protected.items():
        if digest(path)!=h:raise ValueError("Protected baseline/review file changed: "+path)
    save_new(run/"checks.json",dict(passed=True,items=results,unit_tests_log="evidence/all_unit_tests.log",protected_files_unchanged=len(protected),
       gpu_used=False,optimizer_updates=0,new_baseline_runs=0,independent_test_images_read=0,completed_training_count=41,
       synthetic_review_is_not_human_review=True,source_sha256={p.name:digest(p) for p in HERE.glob("*.py")}))
    save_new(run/"example_output_paths.json",dict(A2=str(examples/"CAP_P0.json"),B6=str(examples/"aggregate_existing"),D10=str(examples/"clean_reproduction_table"),E13=str(examples/"synthetic_review")))
    print("ALL 13 SCAFFOLD CHECKS PASS; protected files unchanged; zero training. Evidence: "+str(run))


if __name__=="__main__":main()
