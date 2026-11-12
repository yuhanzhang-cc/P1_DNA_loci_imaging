"""Explicit frozen references; no recursive discovery of possibly synthetic runs."""
from pathlib import Path
from diagnostics.handoff_v2.common import read,sha,write,fresh

BASE=Path("outputs/next_stage_v4_recipe_20261101")
SEEDS=(42,43,44)


def inventory(root,returned=None,scratch=None):
    root=Path(root).resolve();records=[]
    paths={"R0":BASE/"extracted_r0/outputs/research_v4_cuda/baseline_r0/R0_cuda_reference",
        "H0":BASE/"extracted_h0/outputs/research_v4_cuda/baseline_h0/H0_cuda_reference",
        "G0":BASE/"08_returned_gpu/extracted/outputs/research_v4_cuda/recipe/G0_ref_cuda",
        "Spotiflow_F":Path("outputs/next_stage_spotiflow_20261030/06_cuda_results/extracted/outputs/spotiflow_cuda")}
    def add(method,seed,folder,kind,track):
        selection=read(folder/"selection.json");primary=selection["primary"]
        if selection.get("test_evaluated") is not False:raise ValueError("Expected frozen development selection")
        weight=folder/("primary/best.pt" if kind=="spotiflow" else "best_model.pt")
        expected=selection["primary_weight_sha256"] if kind=="spotiflow" else primary["checkpoint_sha256"]
        if sha(weight)!=expected:raise ValueError("Selected reference weight changed: "+method)
        cache=folder/f"epoch_{primary['epoch']:03d}/candidates.npz"
        if not cache.exists():cache=None
        records.append(dict(method=method,seed=seed,kind=kind,track=track,folder=str(folder),
            weight=str(weight),weight_sha256=expected,selection=str(folder/"selection.json"),
            selection_sha256=sha(folder/"selection.json"),threshold=primary["threshold"] if kind=="spotiflow" else primary["calibrated_probability"],
            cache=str(cache) if cache else None,cache_sha256=sha(cache) if cache else None,
            native_config_hashes={n:sha(folder/"primary"/n) for n in ("config.yaml","thresholds.yaml")} if kind=="spotiflow" else {}))
    for method,path in paths.items():
        for seed in SEEDS:add(method,seed,root/path/f"seed_{seed}","spotiflow" if method=="Spotiflow_F" else "own",
            "external_transfer_method_comparison" if method=="Spotiflow_F" else "matched_cuda_reference")
    for method in ("LoG","DoG"):
        folder=root/BASE/"02_classical"/method;selection=read(folder/"selection.json")
        if selection["configurations_completed"]!=24 or selection["test_evaluated"] is not False:raise ValueError("Incomplete classical calibration")
        records.append(dict(method=method,seed=None,kind="classical",track="deterministic_classical",
            folder=str(folder),selection=str(folder/"selection.json"),selection_sha256=sha(folder/"selection.json"),
            cache=str(folder/"selected_candidates.npz"),cache_sha256=sha(folder/"selected_candidates.npz"),
            configuration=selection["configuration"],policy={"kind":selection["primary"]["threshold_kind"],"value":selection["primary"]["threshold"]}))
    if returned:
        from diagnostics.handoff_v2.archive import verify_return
        verify_return(returned)
        for task in read(Path(returned)/"return_manifest.json")["tasks"]:
            add(task["method"],task["seed"],Path(returned)/task["folder"],"signed","prospective_mechanism")
    if scratch:
        # The wrapper returns an intact registered queue and selected caches.
        from .scratch import check_return
        check_return(scratch)
        for seed in SEEDS:add("Spotiflow_scratch",seed,Path(scratch)/"outputs/research_v4_cuda/spotiflow_scratch/Spotiflow_scratch_control"/f"seed_{seed}","spotiflow","external_pretraining_control")
    if len({(r['method'],r['seed']) for r in records})!=len(records):raise ValueError("Duplicate model identity")
    return records


def verify_record(r):
    if sha(r["selection"])!=r["selection_sha256"]:raise ValueError("Selection lock changed")
    if r["kind"]!="classical":
        if sha(r["weight"])!=r["weight_sha256"]:raise ValueError("Weight lock changed")
        choice=read(r["selection"])["primary"]
        value=choice["threshold"] if r["kind"]=="spotiflow" else choice["calibrated_probability"]
        if value!=r["threshold"]:raise ValueError("Locked threshold changed")
        for name,digest in r["native_config_hashes"].items():
            if sha(Path(r["weight"]).parent/name)!=digest:raise ValueError("Spotiflow native config changed")
    if r.get("cache") and sha(r["cache"])!=r["cache_sha256"]:raise ValueError("Complete candidate cache changed")
    return True


def export(root,output,returned=None,scratch=None):
    records=inventory(root,returned,scratch);output=fresh(output)
    write(output/"model_catalog.private.json",records)
    return dict(models=len(records),methods=sorted({r["method"] for r in records}),output=str(output),test_arrays_read=False)


def portable_payload(records,output):
    """Copy only locked selections, selected caches and existing exports."""
    import copy,shutil
    output=fresh(output);records=copy.deepcopy(records)
    for r in records:
        verify_record(r);source=Path(r['folder']);dest=output/'models'/r['method']/(f"seed_{r['seed']}" if r['seed'] is not None else 'deterministic')
        required=[Path(r['selection'])]
        metrics=source/('selected' if r['kind']=='classical' else 'primary')/'metrics.json';required.append(metrics)
        for name in ('prediction_points.csv','matches.csv','per_image_metrics.csv'):
            p=metrics.parent/name
            if p.is_file():required.append(p)
        if r['kind']!='classical':required.append(Path(r['weight']))
        if r.get('cache'):required.append(Path(r['cache']))
        if r['kind']=='spotiflow':required += [Path(r['weight']).parent/n for n in r['native_config_hashes']]
        for p in required:
            q=dest/p.relative_to(source);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
        for key in ('folder','selection','weight','cache'):
            if r.get(key):r[key]=(dest/Path(r[key]).relative_to(source)).relative_to(output).as_posix()
    write(output/'catalog.json',records);return dict(models=len(records),bytes=sum(p.stat().st_size for p in output.rglob('*') if p.is_file()))


def load_portable(folder):
    from diagnostics.handoff_v2.common import inside
    folder=Path(folder);records=read(folder/'catalog.json')
    for r in records:
        for key in ('folder','selection','weight','cache'):
            if r.get(key):r[key]=str(inside(folder,r[key]))
        verify_record(r)
    return records


def live_record(method,seed,folder,kind):
    folder=Path(folder);s=read(folder/'selection.json');p=s['primary'];weight=folder/('primary/best.pt' if kind=='spotiflow' else 'best_model.pt')
    cache=folder/f"epoch_{p['epoch']:03d}/candidates.npz"
    r=dict(method=method,seed=seed,kind=kind,track='external_pretraining_control' if kind=='spotiflow' else 'prospective_mechanism',
        folder=str(folder),selection=str(folder/'selection.json'),selection_sha256=sha(folder/'selection.json'),weight=str(weight),weight_sha256=sha(weight),
        threshold=p['threshold'] if kind=='spotiflow' else p['calibrated_probability'],cache=str(cache),cache_sha256=sha(cache),
        native_config_hashes={n:sha(folder/'primary'/n) for n in ('config.yaml','thresholds.yaml')} if kind=='spotiflow' else {})
    verify_record(r);return r
