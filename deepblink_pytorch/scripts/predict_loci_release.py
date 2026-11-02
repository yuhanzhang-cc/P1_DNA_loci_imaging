"""Fixed-model DNA-locus inference: native 2D TIFF to pixel-coordinate CSV.

This is a reproducible application release, not a new architecture claim.
"""
from pathlib import Path
import argparse
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"deepblink_pytorch"),str(ROOT/"deepblink_pytorch/scripts")]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,help="One 2D TIFF or a folder of TIFF files")
    p.add_argument("--output",type=Path)
    p.add_argument("--method",choices=["spotiflow","g0"],default="spotiflow")
    p.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    p.add_argument("--verify-only",action="store_true")
    args=p.parse_args()
    from research_v4.common import read_json,sha256,dump,write_csv
    manifest=read_json(ROOT/"release_manifest.json")
    for name,digest in manifest["files"].items():
        path=(ROOT/name).resolve()
        if not path.is_relative_to(ROOT.resolve()) or sha256(path)!=digest:raise ValueError("Release integrity mismatch: "+name)
    methods=read_json(ROOT/"method_locks.json")
    if args.verify_only:print(dict(release_verified=True,methods=list(methods),training_performed=False));return
    if args.input is None or args.output is None:p.error("--input and --output are required")
    import numpy as np
    import tifffile
    import torch
    from research_v4.external import locked_candidates
    from research_v4.evaluation import keep_scores
    torch.set_num_threads(4)
    if args.device=="cuda" and not torch.cuda.is_available():raise RuntimeError("Requested CUDA is unavailable; select cpu explicitly if desired")
    source=args.input.resolve();output=args.output.resolve()
    paths=sorted((x for x in source.iterdir() if x.is_file() and x.suffix.lower() in (".tif",".tiff")),key=lambda x:x.name.casefold()) if source.is_dir() else [source]
    if not paths or any(not x.is_file() or x.suffix.lower() not in (".tif",".tiff") for x in paths):raise ValueError("No TIFF inputs")
    if len({x.stem.casefold() for x in paths})!=len(paths):raise ValueError("Input stems collide; preserve unique image identifiers")
    if output==source or output.exists():raise FileExistsError("Use a new output directory; existing data are never overwritten")
    images=[];input_records=[]
    for path in paths:
        native=tifffile.imread(path)
        if native.ndim!=2 or not np.isrealobj(native) or max(native.shape)>512 or min(native.shape)<1:raise ValueError("Require one2D channel with native dimensions<=512: "+str(path))
        image=np.asarray(native,np.float32)
        if not np.isfinite(image).all():raise ValueError("Nonfinite input image")
        images.append(image);input_records.append(dict(file=str(path),sha256=sha256(path),shape=list(image.shape)))
    lock=methods[args.method];threshold=float(lock["threshold"])
    policy=dict(kind="all" if threshold==0 else "none" if threshold==1 else "finite",value=None if threshold in (0,1) else threshold)
    data=dict(x=images,original_shape=np.asarray([x.shape for x in images]))
    output.mkdir(parents=True)
    dump(output/"run_manifest.json",dict(status="running",method=lock,device=args.device,release_manifest_sha256=sha256(ROOT/"release_manifest.json"),
        coordinate_order="y,x",coordinate_unit="pixel",normalization="native Spotiflow auto normalizer" if args.method=="spotiflow" else "native valid-FOV zscore then right/bottom pad512",inputs=input_records))
    start=time.perf_counter()
    try:
        candidates=locked_candidates(lock,ROOT,data,args.device)
        summary=[]
        for path,image,(coords,scores) in zip(paths,images,candidates):
            keep=keep_scores(scores,policy);coords,scores=coords[keep],scores[keep]
            if not np.isfinite(coords).all() or np.any(coords<0) or np.any(coords>=image.shape):raise ValueError("Prediction outside original FOV")
            write_csv(output/"points"/(path.stem+".csv"),[dict(y=float(y),x=float(x),p=float(score)) for (y,x),score in zip(coords,scores)],["y","x","p"])
            summary.append(dict(sample_id=path.stem,height=image.shape[0],width=image.shape[1],n_pred=len(coords),threshold=threshold,method=args.method,coordinate_unit="pixel"))
        write_csv(output/"summary.csv",summary)
        dump(output/"completion.json",dict(status="complete",images=len(paths),predicted_spots=sum(r["n_pred"] for r in summary),elapsed_seconds=time.perf_counter()-start,
             method=args.method,coordinate_unit="pixel",threshold_frozen=True,training_performed=False))
        run=read_json(output/"run_manifest.json");run["status"]="complete";dump(output/"run_manifest.json",run)
        print(read_json(output/"completion.json"))
    except BaseException as error:
        run=read_json(output/"run_manifest.json");run["status"]="failed";dump(output/"run_manifest.json",run)
        dump(output/"failure.json",dict(status="failed",error=repr(error)));raise


if __name__=="__main__":main()
