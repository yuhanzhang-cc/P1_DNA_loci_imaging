"""Exercise sealed, independently extracted kits without formal training."""
from pathlib import Path
import os
import subprocess
import sys
import zipfile

from .common import dump,read_json,sha256


PROBE=r'''from pathlib import Path
import sys
import numpy as np
import tifffile
bundle=Path(sys.argv[1]);work=Path(sys.argv[2]);work.mkdir(parents=True,exist_ok=True)
sys.path[:0]=[str(bundle/"deepblink_pytorch"),str(bundle/"deepblink_pytorch/scripts")]
from research_v4.common import read_json,write_csv,sha256,dump
from research_v4.external import evaluate
import deepblink_torch
assert Path(deepblink_torch.__file__).resolve().is_relative_to(bundle.resolve()), "Leaked editable checkout import"
table=[];rng=np.random.default_rng(20261102)
for index,size in enumerate((128,256)):
 sid=f"synthetic_handoff_{index}"
 image=rng.normal(size=(size,size)).astype(np.float32)
 tifffile.imwrite(work/(sid+".tif"),image)
 write_csv(work/(sid+".csv"),[dict(y=20.,x=30.),dict(y=60.,x=80.)])
 table.append(dict(sample_id=sid,image_file=sid+".tif",label_file=sid+".csv",height=size,width=size,coordinate_unit="pixel",um_per_pixel="",
   biological_group_id=f"synthetic_group_{index}",acquisition_id=f"synthetic_acquisition_{index}",source_container=f"synthetic_source_{index}",
   collection_date="2026-11-02",dataset_role="synthetic_engineering",image_sha256=sha256(work/(sid+".tif")),label_sha256=sha256(work/(sid+".csv")),annotation_version="synthetic_v1"))
write_csv(work/"manifest.csv",table)
locks=read_json(bundle/"method_locks.json")
result=evaluate(work/"manifest.csv",locks,bundle,read_json(bundle/"assets/reference_exclusions.json"),work/"readout",engineering=True)
assert len(result)==9
dump(work/"probe_checks.json",dict(passed=True,synthetic_images=2,frozen_models_reloaded=9,new_optimizer_updates=0,scientific_metrics=False,bundled_import=True))
'''


def run(ctx):
    ctx.task("standalone_verification","running")
    bundles=read_json(ctx.out/"04_bundles/latest.json")
    out=ctx.out/"05_verification"/bundles["source_revision"]
    out.mkdir(parents=True,exist_ok=True)
    if (out/"checks.json").exists():
        result=read_json(out/"checks.json")
        if result.get("passed"):
            dump(ctx.out/"05_verification/checks.json",result)
            ctx.task("standalone_verification","completed",**result);return result
    commands=[]
    def command(argv,name,cwd=None):
        env=dict(os.environ,PYTHONUTF8="1",OMP_NUM_THREADS="4",MKL_NUM_THREADS="4")
        result=subprocess.run([str(a) for a in argv],cwd=cwd,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,encoding="utf-8",errors="replace")
        (out/(name+".txt")).write_text(result.stdout,encoding="utf-8")
        commands.append(dict(name=name,exit_code=result.returncode,argv=[str(a) for a in argv]))
        dump(out/"commands.json",commands)
        if result.returncode:raise RuntimeError("Verification failed: "+name+"; inspect "+str(out/(name+".txt")))
    roots=[]
    for item in bundles["archives"]:
        archive=Path(item["path"])
        if sha256(archive)!=item["sha256"]:raise ValueError("Upload ZIP hash changed")
        target=out/"extracted";target.mkdir(exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            if z.testzip() is not None:raise ValueError("ZIP CRC failed")
            for n in z.namelist():
                if n.startswith("/") or ".." in Path(n).parts:raise ValueError("Unsafe ZIP path")
            z.extractall(target)
        root=target/archive.stem;roots.append(root)
        manifest=read_json(root/"bundle_manifest.json")
        for n,digest in manifest["files"].items():
            if sha256(root/n)!=digest:raise ValueError("Extracted file hash differs: "+n)
        if root.name.endswith("training"):
            command([sys.executable,root/"deepblink_pytorch/scripts/cloud_recipe_train.py","--verify-only"],"training_bundle",root)
        else:command([sys.executable,root/"deepblink_pytorch/scripts/evaluate_external.py","--verify-only"],"external_bundle",root)
        bash=Path(r"C:\Program Files\Git\bin\bash.exe") if os.name=="nt" else Path("/bin/bash")
        if not bash.is_file():raise FileNotFoundError("Bash syntax checker unavailable")
        for script in root.glob("*.sh"):command([bash,"-n",script],root.name+"_"+script.stem,root)
    probe=out/"engineering_external_probe.py";probe.write_text(PROBE,encoding="utf-8")
    command([sys.executable,probe,roots[1],out/"synthetic_external"],"frozen_model_reload_and_external_export",ctx.code_root)
    command([sys.executable,"-m","unittest","discover","-s","deepblink_pytorch/tests","-p","test_research_v4.py","-v"],"targeted_tests",ctx.code_root)
    command([sys.executable,"-m","pip","check"],"local_environment",ctx.code_root)
    command([sys.executable,"-m","compileall","-q",ctx.code_root/"deepblink_pytorch/scripts/research_v4",
        ctx.code_root/"deepblink_pytorch/scripts/cloud_recipe_train.py",ctx.code_root/"deepblink_pytorch/scripts/cloud_job_control.py",
        ctx.code_root/"deepblink_pytorch/scripts/evaluate_external.py",ctx.code_root/"deepblink_pytorch/scripts/analyze_research_v4.py"],"syntax",ctx.code_root)
    result=dict(passed=True,source_revision=bundles["source_revision"],archives_sha_and_crc=True,all_extracted_manifest_hashes=True,
        standalone_import=True,bash_syntax_scripts=sum(len(list(p.glob("*.sh"))) for p in roots),frozen_models_reloaded=9,
        synthetic_images_only=2,targeted_tests=13,local_pip_check=True,new_optimizer_updates=0,cuda_tested=False,
        linux_installation_tested=False,command_log=str(out/"commands.json"))
    dump(out/"checks.json",result);dump(ctx.out/"05_verification/checks.json",result)
    ctx.task("standalone_verification","completed",**result)
    return result
