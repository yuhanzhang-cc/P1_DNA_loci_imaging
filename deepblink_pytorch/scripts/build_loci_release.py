"""Build a two-model inference candidate from already frozen selected weights."""
from pathlib import Path
import argparse
import shutil
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"deepblink_pytorch"),str(ROOT/"deepblink_pytorch/scripts")]
from research_v4.common import dump,read_json,sha256,rows


README="""# P1 DNA-locus detection application candidate, 2026-11-02

Default model: Spotiflow general fine-tuned on the frozen220-image study, fixed seed42 / epoch60 / threshold0.6553990357712657. Development validation micro-F1=0.8600 on22 images,372 points,3px matching. Three-seed study mean was0.8606 +/-0.0034 sample SD. Model weights come from the existing completed study.

Alternative: G0 simple internal reference, fixed seed42 / epoch65 / threshold0.01098694263059318; original XPU-trained checkpoint, development micro-F1=0.8261. The new CUDA recipe experiment is separately archived; joint augmentation did not improve mean3px F1.

The current deliverable is a fixed-model detection/localization tool and an evidence report. Core architecture contribution and independent biological validation remain research gates. Spotiflow is an external pretrained/locally adapted method; G0 is an internal simple reference.

## Inference

Use Python3.12 and requirements-research-v4.txt with torch2.11.0; the existing p1_v4_training .venv_cuda or local project Spotiflow environment can be reused. Put new input images outside this release directory and select a new output directory for each run.

```bash
python deepblink_pytorch/scripts/predict_loci_release.py --verify-only
python deepblink_pytorch/scripts/predict_loci_release.py --input /path/to/tifs --output /path/to/results --device cuda
python deepblink_pytorch/scripts/predict_loci_release.py --input /path/to/image.tif --output /path/to/g0_result --method g0 --device cpu
```

Inputs: native single-channel2D TIFF, each dimension<=512. Multichannel/Z-stack/channel selection requires explicit upstream preparation. TIFF pixels are preserved. G0 normalizes the native valid region and pads on right/bottom to512; Spotiflow uses the frozen native auto-normalizer/subpixel/NMS adapter. Predictions remain in the original image coordinates.

Outputs: one points/IMAGE.csv with y,x,p columns per image, summary.csv, run_manifest.json and completion.json. Coordinates use pixels. Physical calibration remains external metadata; no micron scale is guessed. The threshold is frozen per model; inference performs no calibration or training. Existing output directories are preserved. Method/weight/configuration hashes and input hashes are logged.

Archive SHA256 and release_manifest.json cover all delivered files. Verification on the current host is recorded separately by the builder. Recheck the actual CUDA environment before production inference on a new server.

## Source methods

Spotiflow: Dominguez Mantes et al., Nature Methods22,1495-1504(2025), https://doi.org/10.1038/s41592-025-02662-x .
deepBlink: Eichenberger et al., Nucleic Acids Research49,7292-7297(2021), https://doi.org/10.1093/nar/gkab546 . G0 is a project control built on the PyTorch migration, not an official pretrained deepBlink model.
"""


def copy(source,target):
    target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--data-root",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();data=args.data_root.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    bundle=out/"p1_loci_detection_rc_20261102"
    if bundle.exists():raise FileExistsError("Preserve the sealed release; select a new output revision")
    bundle.mkdir()
    selected=list((ROOT/"deepblink_pytorch/deepblink_torch").rglob("*.py"))
    selected+=[ROOT/"deepblink_pytorch/scripts/predict_loci_release.py"]
    for n in ("__init__.py","common.py","augmentation.py","evaluation.py","engine.py","external.py"):
        selected.append(ROOT/"deepblink_pytorch/scripts/research_v4"/n)
    for n in ("pyproject.toml","README.md","requirements-research-v4.txt"):selected.append(ROOT/"deepblink_pytorch"/n)
    for source in selected:copy(source,bundle/source.relative_to(ROOT))
    own=data/"outputs/next_stage_v3_closing_controls/01_screen/G0/seed_42";selection=read_json(own/"selection.json")
    relative="models/g0/best_model.pt";copy(own/"best_model.pt",bundle/relative)
    if sha256(bundle/relative)!=selection["checkpoint_sha256"]:raise ValueError("G0 checkpoint changed")
    own_lock=dict(method="G0",seed=42,kind="own",checkpoint=relative,weight_sha256=sha256(bundle/relative),threshold=selection["primary"]["threshold"],
        pretrained=False,configuration_hashes={},source_role="frozen historical simple reference; XPU trained")
    source=data/"outputs/next_stage_spotiflow_20261030/06_cuda_results/extracted/outputs/spotiflow_cuda/seed_42"
    selection=read_json(source/"selection.json");relative="models/spotiflow"
    for n in ("best.pt","config.yaml","thresholds.yaml"):copy(source/"primary"/n,bundle/relative/n)
    if sha256(bundle/relative/"best.pt")!=selection["primary_weight_sha256"]:raise ValueError("Spotiflow checkpoint changed")
    external_lock=dict(method="Spotiflow_F",seed=42,kind="spotiflow",checkpoint=relative,weight_sha256=sha256(bundle/relative/"best.pt"),
        threshold=selection["primary"]["threshold"],pretrained=True,configuration_hashes={n:sha256(bundle/relative/n) for n in ("config.yaml","thresholds.yaml")},
        source_role="frozen adapted external method; CUDA4090 trained")
    dump(bundle/"method_locks.json",dict(spotiflow=external_lock,g0=own_lock))
    (bundle/"README.md").write_text(README,encoding="utf-8")
    manifest={p.relative_to(bundle).as_posix():sha256(p) for p in sorted(bundle.rglob("*")) if p.is_file()}
    dump(bundle/"release_manifest.json",dict(release_id=bundle.name,files=manifest,default_method="spotiflow",data_included=False,
        architecture_innovation_status="not_established",independent_validation_status="pending",new_training_tasks=0))
    validation=out/"verification";validation.mkdir(exist_ok=True)
    commands=[]
    for method in ("spotiflow","g0"):
        command=[sys.executable,str(bundle/"deepblink_pytorch/scripts/predict_loci_release.py"),"--input",str(data/"data/raw/tif/sample_111.tif"),
                 "--output",str(validation/method),"--method",method,"--device","cpu"]
        result=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,encoding="utf-8",errors="replace")
        (validation/(method+".log")).write_text(result.stdout,encoding="utf-8")
        if result.returncode:raise RuntimeError("Release inference failed: "+str(validation/(method+".log")))
        if read_json(validation/method/"completion.json")["status"]!="complete":raise ValueError("Incomplete output")
        commands.append(dict(method=method,passed=True,coordinate_unit="pixel",new_optimizer_updates=0,
                             output=read_json(validation/method/"completion.json")))
    archive=out/(bundle.name+".zip")
    with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=5) as z:
        for name in [*manifest,"release_manifest.json"]:z.write(bundle/name,(Path(bundle.name)/name).as_posix())
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise ValueError("Release ZIP CRC failed")
        if any("__pycache__" in n or ".venv" in n or n.endswith((".tif",".tiff",".npz")) for n in z.namelist()):raise ValueError("Unexpected data/cache in release")
    result=dict(passed=True,release_id=bundle.name,archive=str(archive),archive_sha256=sha256(archive),bytes=archive.stat().st_size,
                delivered_files=len(manifest)+1,models=2,default_method="spotiflow",cpu_inference_checks=commands,cuda_inference_tested_here=False,
                source_builder_sha256=sha256(Path(__file__)),raw_images_included=False,new_training_tasks=0)
    archive.with_name(archive.name+".sha256").write_text(result["archive_sha256"]+"  "+archive.name+"\n",encoding="ascii")
    dump(out/"release_checks.json",result);print(result)


if __name__=="__main__":main()
