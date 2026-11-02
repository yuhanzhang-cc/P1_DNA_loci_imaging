"""Portable manifest-bound CUDA queues, with no CPU fallback for formal work."""
from pathlib import Path
import argparse
import os
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
os.environ.setdefault("OMP_NUM_THREADS","4")
os.environ.setdefault("MKL_NUM_THREADS","4")
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"deepblink_pytorch"))


def verify_bundle():
    from research_v4.common import read_json,sha256,Corpus,load_spec
    manifest=read_json(ROOT/"bundle_manifest.json")
    for name,digest in manifest["files"].items():
        p=ROOT/name
        if not p.resolve().is_relative_to(ROOT.resolve()) or not p.is_file() or sha256(p)!=digest:raise ValueError("Bundle identity/hash mismatch: "+name)
    spec=load_spec(ROOT/"protocol_proposed.yaml")
    corpus=Corpus(ROOT,spec,assets_root=ROOT/"assets")
    return manifest,spec,corpus


def check_complete(output,queue):
    from research_v4.common import read_json,sha256
    from research_v4.queues import queues
    config=read_json(output/queue/"protocol.json")
    if config["queue"]!=queue or config["tasks"]!=queues()[queue]:raise ValueError("Completed queue task registry differs")
    for item in queues()[queue]:
        folder=output/queue/item["task_id"]/f"seed_{item['seed']}"
        if not (folder/"status.json").exists() or read_json(folder/"status.json")["status"]!="complete":raise ValueError("Queue is incomplete: "+str(folder))
        if read_json(folder/"task_protocol.json")!=dict(item=item,queue=config):raise ValueError("Completed task has no matching registration")
        selection=read_json(folder/"selection.json")
        if selection["checkpoint_count"]!=20 or selection["epoch_threshold_pairs"]!=620:raise ValueError("Formal selection budget is incomplete")
        if item["pipeline"]=="own":
            registration=read_json(folder/"registered_config.json");status=read_json(folder/"status.json")
            if registration["engineering_only"] or registration["device"]!="cuda" or registration["source_lock"]!=config["manifest_sha256"]:raise ValueError("Engineering or foreign-source result cannot complete a formal task")
            if status["epoch"]!=100 or status["updates"]!=2200:raise ValueError("Formal epoch/update budget incomplete")
            path,digest=folder/"best_model.pt",selection["primary"]["checkpoint_sha256"]
            if sha256(folder/"native_selected_model.pt")!=selection["native_secondary"]["checkpoint_sha256"]:raise ValueError("Secondary selected model changed")
        else:
            budget=read_json(folder/"optimizer_summary.json")
            if budget["completed_epochs"]!=100 or budget["optimizer_updates"]!=4400:raise ValueError("Native formal epoch/update budget incomplete")
            path,digest=folder/"primary/best.pt",selection["primary_weight_sha256"]
        if sha256(path)!=digest:raise ValueError("Completed model changed")
        metrics=read_json(folder/"primary/metrics.json")
        for key in ("tp","fp","fn","f1"):
            if metrics[key]!=selection["primary"][key]:raise ValueError("Selected metrics differ from export")


def queue_probe(root,queue,corpus,device):
    """Probe the actual adapter geometry; no formal training results are retained."""
    import numpy as np
    import torch
    from research_v4.engine import configure_device,reset_rng,make_model,forward
    from research_v4.augmentation import prepare_sample
    from deepblink_torch.losses import combined_dice_rmse_loss
    from deepblink_torch.round2_protocol import dense_heatmap_loss
    configure_device(device);reset_rng(42)
    data=corpus.splits["train"]
    indices=[i for i,s in enumerate(data["original_shape"]) if int(s[0])==512][:8 if device=="cuda" else 1]
    if queue=="spotiflow_scratch":
        if device!="cuda":raise ValueError("Native full-geometry queue probe requires actual CUDA")
        from research_v4.native_adapter import scratch_model
        from spotiflow_overnight.engine import train_native,default_training
        model=scratch_model(ROOT/"assets/spotiflow_config.yaml",42)
        config=default_training(epochs=1,batch_size=4,crop_size=256);config.finetuned_from=None
        train_native(model,[data["x"][i] for i in indices[:4]],[data["y"][i] for i in indices[:4]],
                     corpus.splits["valid"]["x"][:1],corpus.splits["valid"]["y"][:1],root/"scratch_probe",seed=42,config=config,device=device)
        return dict(passed=True,adapter=queue,discarded_real_optimizer_updates=1,batch_size=4,crop_size=256)
    key={"recipe":"G0","baseline_r0":"R0","baseline_h0":"H0"}[queue]
    model=make_model(key,device)
    samples=[prepare_sample(data["x"][i],data["y"][i],dense=key=="H0") for i in indices]
    x=torch.from_numpy(np.stack([a[0] for a in samples])).to(device)
    y=torch.from_numpy(np.stack([a[1] for a in samples])).to(device)
    prediction=forward(model,x,y)
    loss=(dense_heatmap_loss if key=="H0" else combined_dice_rmse_loss)(y,prediction)
    loss.backward()
    if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError("Adapter probe is nonfinite")
    return dict(passed=True,adapter=queue,batch_size=len(indices),canvas=512,finite_loss=float(loss.detach()),discarded_optimizer_updates=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--verify-only",action="store_true")
    mode.add_argument("--smoke",action="store_true")
    mode.add_argument("--train",action="store_true")
    mode.add_argument("--status",action="store_true")
    mode.add_argument("--check-complete",action="store_true")
    mode.add_argument("--ready-check",action="store_true")
    parser.add_argument("--queue",choices=["recipe","baseline_r0","baseline_h0","spotiflow_scratch"],default="recipe")
    parser.add_argument("--device",choices=["cpu","cuda"],default="cuda")
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--output",type=Path,default=ROOT/"outputs/research_v4_cuda")
    args=parser.parse_args()
    import torch
    from research_v4.common import dump,read_json,sha256,write_csv
    from research_v4.queues import queues,validate
    from research_v4.engine import smoke,train
    from filelock import FileLock
    manifest,spec,corpus=verify_bundle()
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    if args.train and not output.is_relative_to(ROOT.resolve()):raise ValueError("Formal output must remain inside the portable bundle for model locks")
    if args.verify_only:
        print(dict(bundle_verified=True,train_images=176,valid_images=22,test_arrays_present=False,queues={k:len(v) for k,v in queues().items()}));return
    if args.status:
        for queue,items in queues().items():
            for item in items:
                p=output/queue/item["task_id"]/f"seed_{item['seed']}"
                status=read_json(p/"status.json") if (p/"status.json").exists() else dict(status="not_started")
                progress=read_json(p/"progress.json") if (p/"progress.json").exists() else {}
                print(queue,item["task_id"],item["seed"],status["status"],f"epoch {progress.get('epoch',status.get('epoch',0))}/100")
        return
    if args.check_complete:check_complete(output,args.queue);print("All registered tasks complete:",args.queue);return
    torch.set_num_threads(4)
    if args.smoke:
        result=smoke(output/"engineering_smoke",device=args.device,corpus=corpus)
        binding=dict(manifest_sha256=sha256(ROOT/"bundle_manifest.json"),device=args.device,torch=str(torch.__version__))
        dump(output/"engineering_smoke/bundle_binding.json",binding)
        probe_path=output/"engineering_smoke"/(args.queue+"_probe.json")
        if probe_path.exists():
            probe=read_json(probe_path)
            if probe["binding"]!=binding:raise ValueError("Adapter smoke belongs to another bundle")
        else:
            probe=dict(binding=binding,**queue_probe(output/"engineering_smoke",args.queue,corpus,args.device))
            dump(probe_path,probe)
        print(result);return
    if args.device!="cuda" or not torch.cuda.is_available():raise RuntimeError("Formal queues require CUDA; no CPU fallback")
    marker=read_json(output/"engineering_smoke/checks.json")
    binding=read_json(output/"engineering_smoke/bundle_binding.json")
    if not marker.get("passed") or marker.get("device")!="cuda" or marker.get("torch_version")!=str(torch.__version__) or not marker.get("cuda_tested"):
        raise ValueError("Actual CUDA smoke is required")
    if binding!=dict(manifest_sha256=sha256(ROOT/"bundle_manifest.json"),device="cuda",torch=str(torch.__version__)):raise ValueError("GPU smoke belongs to another bundle/environment")
    probe=read_json(output/"engineering_smoke"/(args.queue+"_probe.json"))
    if probe.get("binding")!=binding or not probe.get("passed"):raise ValueError("This queue needs its own actual CUDA adapter smoke")
    if args.ready_check:print("CUDA queue gates passed:",args.queue);return
    items=queues()[args.queue];validate(args.queue,items)
    folder=output/args.queue;folder.mkdir(exist_ok=True)
    config=dict(queue=args.queue,tasks=items,device="cuda",torch=str(torch.__version__),gpu=torch.cuda.get_device_name(0),
                manifest_sha256=sha256(ROOT/"bundle_manifest.json"),test_arrays_loaded=False)
    if (folder/"protocol.json").exists() and read_json(folder/"protocol.json")!=config:raise ValueError("Queue protocol changed")
    with FileLock(str(output/".gpu_runner.lock"),timeout=0):
        dump(folder/"protocol.json",config)
        dump(folder/"controller_status.json",dict(status="running",pid=os.getpid(),queue=args.queue,maximum_tasks=len(items)))
        try:
            for item in items:
                p=folder/item["task_id"]/f"seed_{item['seed']}"
                if (p/"task_protocol.json").exists() and read_json(p/"task_protocol.json")!=dict(item=item,queue=config):raise ValueError("Task protocol differs")
                if (p/"status.json").exists() and read_json(p/"status.json")["status"]=="complete":
                    if not args.resume:raise ValueError("Existing task requires audited --resume")
                    if not (p/"task_protocol.json").exists():raise ValueError("Cannot skip an unregistered completed task")
                    continue
                dump(p/"task_protocol.json",dict(item=item,queue=config))
                if item["pipeline"]=="own":
                    train(corpus.splits,p,key=item["key"],augment=item["augment"],seed=item["seed"],device="cuda",resume=args.resume and (p/"recovery.pt").exists(),
                          source_lock=config["manifest_sha256"],event=lambda event,**fields:print(__import__("json").dumps(dict(event=event,**fields)),flush=True))
                else:
                    from research_v4.native_adapter import train_scratch
                    train_scratch(corpus.splits,p,ROOT/"assets/spotiflow_config.yaml",seed=item["seed"],device="cuda",resume=args.resume and (p/"recovery/last.ckpt").exists())
            check_complete(output,args.queue)
            records=[];locks=[]
            for item in items:
                p=folder/item["task_id"]/f"seed_{item['seed']}";selection=read_json(p/"selection.json");choice=selection["primary"]
                metrics=choice["metrics"] if item["pipeline"]=="own" else read_json(p/"primary/metrics.json")
                checkpoint=p/("best_model.pt" if item["pipeline"]=="own" else "primary")
                weight=checkpoint if checkpoint.is_file() else checkpoint/"best.pt"
                threshold=choice["calibrated_probability"] if item["pipeline"]=="own" else choice["threshold"]
                records.append(dict(task_id=item["task_id"],seed=item["seed"],epoch=choice["epoch"],threshold=threshold,**{k:metrics[k] for k in ("tp","fp","fn","precision","recall","f1","weak_recall")}))
                locks.append(dict(method=item["task_id"],seed=item["seed"],kind="own" if item["pipeline"]=="own" else "spotiflow",checkpoint=str(checkpoint.relative_to(ROOT)),
                                  weight_sha256=sha256(weight),threshold=threshold,pretrained=False,
                                  configuration_hashes={} if item["pipeline"]=="own" else {n:sha256(checkpoint/n) for n in ("config.yaml","thresholds.yaml")}))
            write_csv(folder/"summary_per_seed.csv",records);dump(folder/"method_locks.json",locks)
            dump(folder/"controller_status.json",dict(status="complete",queue=args.queue,completed_tasks=len(items),test_evaluated=False))
        except BaseException as error:
            (folder/"failure.txt").write_text(__import__("traceback").format_exc(),encoding="utf-8")
            dump(folder/"controller_status.json",dict(status="failed_or_interrupted",error=repr(error),queue=args.queue));raise


if __name__=="__main__":
    if hasattr(sys.stdout,"reconfigure"):sys.stdout.reconfigure(encoding="utf-8")
    main()
