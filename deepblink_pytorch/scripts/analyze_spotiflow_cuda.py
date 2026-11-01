"""Audit returned CUDA artifacts and compare frozen development predictions.

No training, test-array access, threshold refinement, or annotation changes.
Run with the existing spotiflow_cpu environment; see --help for input roots.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import scipy
import torch
import yaml

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
from deepblink_torch.evaluation import match_coordinates
from deepblink_torch.round2_protocol import THRESHOLDS_V3, threshold_mask
from deepblink_torch.spotiflow_bridge import SPLIT_SHA256, evaluate_checkpoints, read_candidates


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def table(path, records):
    assert records
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def counts_metrics(counts):
    tp, fp, fn = np.moveaxis(np.asarray(counts, dtype=float), -1, 0)
    precision = np.divide(tp, tp + fp, out=np.asarray(tp + fn == 0, dtype=float), where=tp + fp > 0)
    recall = np.divide(tp, tp + fn, out=np.ones_like(tp), where=tp + fn > 0)
    f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.ones_like(tp), where=2 * tp + fp + fn > 0)
    return dict(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1)


def exact_seed_test(delta):
    """Exploratory two-sided sign-flip test; only eight assignments for n=3."""
    delta = np.asarray(delta)
    null = np.asarray([np.mean(delta * signs) for signs in itertools.product((-1, 1), repeat=len(delta))])
    return float(np.mean(np.abs(null) >= abs(delta.mean()) - 1e-12))


def holm(p_values):
    order = np.argsort(p_values)
    result = np.zeros(len(order))
    highest = 0.
    for rank, index in enumerate(order):
        highest = max(highest, (len(order) - rank) * p_values[index])
        result[index] = min(highest, 1.)
    return result


def audit_predictions(folder, data, weak, *, model, seed, role, choice):
    predictions = read_csv(folder / "prediction_points.csv")
    saved_images = {r["sample_id"]: r for r in read_csv(folder / "per_image_metrics.csv")}
    assert set(saved_images) == set(data["sample_id"])
    assert set(r["sample_id"] for r in predictions) <= set(data["sample_id"])
    per_image, matched_truth, distances = [], {}, []
    weak_tp = 0
    for sid, shape, truth in zip(data["sample_id"], data["original_shape"], data["y"]):
        records = [r for r in predictions if r["sample_id"] == sid]
        coords = np.asarray([[float(r["y"]), float(r["x"])] for r in records], dtype=np.float32).reshape(-1, 2)
        assert np.isfinite(coords).all() and np.all(coords >= 0) and np.all(coords < shape)
        match = match_coordinates(coords, truth, 3.)
        counts = np.asarray([match.tp, match.fp, match.fn])
        for key, value in zip(("tp", "fp", "fn"), counts):
            assert int(float(saved_images[sid][key])) == value, (model, seed, sid, key)
        matched_truth[sid] = {int(i): float(d) for i, d in zip(match.matched_true_indices, match.matched_distances)}
        weak_tp += sum(weak[(sid, int(i))] for i in match.matched_true_indices)
        distances.extend(map(float, match.matched_distances))
        metrics = counts_metrics(counts)
        per_image.append(dict(model=model, seed=seed, role=role, sample_id=sid, size=int(shape[0]),
                              **{k: float(v) for k, v in metrics.items()}))
    total = np.asarray([[r[k] for k in ("tp", "fp", "fn")] for r in per_image]).sum(0)
    metrics = {k: float(v) for k, v in counts_metrics(total).items()}
    for key in ("tp", "fp", "fn", "precision", "recall", "f1"):
        assert np.isclose(metrics[key], choice[key], rtol=0, atol=1e-10), (model, seed, key)
    result = dict(model=model, seed=seed, role=role, epoch=int(choice["epoch"]), threshold=float(choice["threshold"]),
                  **metrics, weak_tp=int(weak_tp), weak_n=96, weak_recall=weak_tp / 96,
                  mean_error_px=float(np.mean(distances)), median_error_px=float(np.median(distances)),
                  rmse_error_px=float(np.sqrt(np.mean(np.square(distances)))),
                  macro_f1=float(np.mean([r["f1"] for r in per_image])),
                  count_mae=float(np.mean([abs(r["fp"] - r["fn"]) for r in per_image])))
    return result, per_image, matched_truth


def deployment_probe(cloud, bundle, data, root):
    """Reload all three native models and check one fixed image per geometry."""
    from spotiflow.model import Spotiflow
    from deepblink_torch.spotiflow_bridge import spotiflow_candidates
    from deepblink_torch.networks.closing_controls import UNetDeepLociGridControl
    torch.set_num_threads(4)
    with np.load(bundle / "assets/development_only.npz", allow_pickle=True) as archive:
        images = archive["x_valid"]
    indices = [next(i for i, shape in enumerate(data["original_shape"]) if shape[0] == size) for size in (256,512)]
    result = []
    for seed in (42,43,44):
        folder = cloud / f"seed_{seed}/primary"
        model = Spotiflow.from_folder(folder, inference_mode=True)
        parameter_count = sum(p.numel() for p in model.parameters())
        threshold = yaml.safe_load((folder/"thresholds.yaml").read_text())["prob_thresh_best"][0]
        saved = read_csv(folder/"prediction_points.csv")
        for i in indices:
            coords, scores = spotiflow_candidates(model, images[i], "cpu")
            keep = threshold_mask(scores, threshold)
            coords, scores = coords[keep], scores[keep]
            old = [r for r in saved if r["sample_id"] == data["sample_id"][i]]
            old_coords = np.asarray([[float(r["y"]),float(r["x"])] for r in old],dtype=np.float32).reshape(-1,2)
            old_scores = np.asarray([float(r["p"]) for r in old])
            matched = match_coordinates(coords, old_coords, .01)
            assert matched.tp == len(coords) == len(old_coords), (seed,i,"CPU/GPU selected predictions differ")
            delta = np.abs(scores[matched.matched_pred_indices]-old_scores[matched.matched_true_indices])
            assert np.max(delta,initial=0)<1e-3
            result.append(dict(seed=seed,size=int(data["original_shape"][i][0]),selected_points=len(coords),
                               cpu_gpu_max_position_difference_px=float(np.max(matched.matched_distances,initial=0)),
                               cpu_gpu_max_score_difference=float(np.max(delta,initial=0)),passed=True))
        del model
        print(f"seed {seed}: reloaded model reproduces CUDA predictions on fixed 256/512 images",flush=True)
    control = UNetDeepLociGridControl()
    control.load_state_dict(torch.load(root/"outputs/next_stage_v3_closing_controls/01_screen/G0/seed_42/best_model.pt",
                                      map_location="cpu",weights_only=True)["model_state_dict"])
    return result,dict(G0=sum(p.numel() for p in control.parameters()),Spotiflow_F=parameter_count)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--publish-root", required=True, type=Path)
    parser.add_argument("--probe-deployment", action="store_true", help="Also reload three models on CPU and compare six fixed-image predictions")
    args = parser.parse_args()
    root, publish = args.project_root.resolve(), args.publish_root.resolve()
    parent = root / "outputs/next_stage_spotiflow_20261030"
    cloud = parent / "06_cuda_results/extracted/outputs/spotiflow_cuda"
    bundle = parent / "05_cuda_package/bundle"
    out = parent / "07_cuda_analysis"
    out.mkdir(exist_ok=True)
    paper = publish / "docs/next_stage_plan_20261025/cuda_analysis_20261030"
    paper.mkdir(exist_ok=True)
    # Plan fixed before recomputation. The two contrasts are exploratory, not a new test protocol.
    plan = dict(dataset="same 22 development-validation images / 372 reference points", seeds=[42, 43, 44],
                contrasts=["Spotiflow_F-G0", "Spotiflow_F-R0"], family_size=2,
                seed_test="two-sided exact sign flips; Holm across two contrasts; descriptive n=3",
                bootstrap="10000 paired image draws, seed 20261025, same indices across methods and seeds",
                bootstrap_scope="conditional on selected models and current validation images; not selection-adjusted or biological CI",
                forbidden=["test-array access", "retraining", "new threshold grid", "relabeling"],
                figure_scope="general research progress report, no journal compliance claim")
    dump(out / "analysis_plan.json", plan)
    audit = {"test_arrays_loaded": False, "server_sha256_available": False, "checks": {}}
    protocol = read_json(cloud / "protocol.json")
    smoke = read_json(cloud / "engineering_smoke/checks.json")
    manifest_path = cloud.parents[1] / "bundle_manifest.json"
    manifest = read_json(manifest_path)
    assert protocol["device"] == "cuda" and protocol["seeds"] == [42, 43, 44]
    assert smoke["passed"] and smoke["cuda_tested"] and smoke["device"] == "cuda"
    assert protocol["bundle_manifest_sha256"] == sha256(manifest_path) == sha256(bundle / "bundle_manifest.json")
    assert not manifest["test_arrays_present"] and not protocol["test_arrays_loaded"]
    dev_path = bundle / "assets/development_only.npz"
    assert sha256(dev_path) == manifest["files"]["assets/development_only.npz"]
    expected = {f"{k}_{s}" for k in ("x", "y", "sample_id", "original_shape") for s in ("train", "valid")}
    with np.load(dev_path, allow_pickle=True) as archive:
        assert set(archive.files) == expected
        data = {k: archive[f"{k}_valid"] for k in ("sample_id", "original_shape", "y")}
        train_ids = set(map(str, archive["sample_id_train"]))
        train_shapes = archive["original_shape_train"]
    data["sample_id"] = np.asarray(data["sample_id"], dtype=str)
    assert len(data["sample_id"]) == 22 and sum(map(len, data["y"])) == 372
    assert len(train_ids) == 176 and not train_ids.intersection(data["sample_id"])
    assert not ({"sample_230", "sample_286"} & (train_ids | set(data["sample_id"])))
    split_path = root/"outputs/261024_clean220_random/frozen_split.csv"
    assert sha256(split_path) == SPLIT_SHA256
    split_rows = read_csv(split_path)
    containers = {subset: {r["source_group"].split("|")[0] for r in split_rows if r["subset"]==subset}
                  for subset in ("train","valid","test")}
    audit["source_container_audit"] = dict(total=len(set.union(*containers.values())),
        train=len(containers["train"]),valid=len(containers["valid"]),
        train_valid_overlap=len(containers["train"] & containers["valid"]),
        interpretation="container identity is metadata, not verified biological independence")
    features = read_csv(root / "outputs/next_stage_v3_closing_controls/02_analysis/validation_truth_features.csv")
    weak = {(r["sample_id"], int(r["true_index"])): r["weak"] == "True" for r in features}
    assert len(weak) == 372 and sum(weak.values()) == 96
    for sid, truth in zip(data["sample_id"], data["y"]):
        stored = [r for r in features if r["sample_id"] == sid]
        assert np.allclose([[float(r["y"]), float(r["x"])] for r in stored], truth, atol=1e-6)
    lock = read_json(parent / "00_protocol/lock.json")
    for name, digest in lock["protected_references"].items():
        assert sha256(root / name) == digest, "Protected historical artifact changed: " + name
    audit["checks"]["protected_historical_artifacts"] = len(lock["protected_references"])
    audit["checks"]["bundle_manifest_and_dev_split"] = True
    audit["checks"]["cuda_smoke"] = True
    audit["input_hashes"] = {"bundle_manifest": sha256(manifest_path), "development_only": sha256(dev_path),
                             "returned_archive": read_json(parent / "06_cuda_results/local_extraction_check.json")["local_sha256"]}
    records, images, truth_matches, learning, timing, grid_records = [], [], {}, [], [], []
    for model, study in (("R0", "next_stage_v3"), ("G0", "next_stage_v3_closing_controls"),
                         ("Qglobal", "next_stage_v3_closing_controls")):
        for seed in (42, 43, 44):
            folder = root / f"outputs/{study}/01_screen/{model}/seed_{seed}"
            choice = read_json(folder / "selection.json")["primary"]
            result, per_image, matches = audit_predictions(folder, data, weak, model=model, seed=seed, role="primary", choice=choice)
            records.append(result); images.extend(per_image); truth_matches[(model, seed)] = matches
    for seed in (42, 43, 44):
        folder = cloud / f"seed_{seed}"
        selection, snapshots = read_json(folder / "selection.json"), read_json(folder / "snapshots.json")
        pipeline, optimizer = read_json(folder / "resolved_pipeline.json"), read_json(folder / "optimizer_summary.json")
        assert read_json(folder / "status.json")["status"] == "complete"
        assert pipeline["training_images"] == 176 and pipeline["validation_images"] == 22
        assert pipeline["device"] == "cuda" and pipeline["precision"] == "32-true"
        assert optimizer["completed_epochs"] == 100 and optimizer["optimizer_updates"] == 4400
        assert [s["epoch"] for s in snapshots] == list(range(5, 101, 5))
        assert selection["checkpoint_count"] == 20 and selection["epoch_threshold_pairs"] == 620
        candidates = []
        for item in snapshots:
            epoch = item["epoch"]; path = folder / f"epoch_{epoch:03d}"
            assert sha256(path / "best.pt") == item["weight_sha256"]
            assert sha256(path / "candidates.npz") == item["candidate_sha256"]
            candidates.append((epoch, read_candidates(path / "candidates.npz", data)))
            learning.append(dict(seed=seed, epoch=epoch, native_val_loss=item["native_val_loss"], **{
                "best_fullimage_f1": item["best_at_epoch"]["f1"],
                "best_threshold": item["best_at_epoch"]["threshold"]}))
        assert len({s["weight_sha256"] for s in snapshots}) == 20
        recalculated, selected, _ = evaluate_checkpoints(candidates, data)
        saved_rows = read_csv(folder / "all_epoch_threshold_metrics.csv")
        assert len(saved_rows) == len(recalculated) == 620
        for actual, stored in zip(recalculated, saved_rows):
            for key, value in actual.items():
                assert np.isclose(value, float(stored[key]), rtol=0, atol=1e-10), (seed, key)
        assert selected == selection["primary"]
        native_epoch = min(snapshots, key=lambda s: (s["native_val_loss"], s["epoch"]))["epoch"]
        secondary = max([r for r in recalculated if r["epoch"] == native_epoch], key=lambda r: (r["f1"], r["precision"], r["threshold"]))
        assert secondary == selection["native_secondary"]
        grid_records.extend(dict(seed=seed, **r) for r in recalculated)
        for role, label in (("primary", "Spotiflow_F"), ("native_secondary", "Spotiflow_native_secondary")):
            choice = selection[role]
            assert sha256(folder / role / "best.pt") == sha256(folder / f"epoch_{choice['epoch']:03d}/best.pt")
            threshold = yaml.safe_load((folder / role / "thresholds.yaml").read_text())["prob_thresh_best"][0]
            assert threshold == choice["threshold"]
            result, per_image, matches = audit_predictions(folder / role, data, weak, model=label, seed=seed, role=role, choice=choice)
            saved_metrics = read_json(folder / role / "metrics.json")
            for key in ("tp", "fp", "fn", "f1", "weak_recall"):
                assert np.isclose(result[key], saved_metrics[key], atol=1e-10)
            records.append(result); images.extend(per_image); truth_matches[(label, seed)] = matches
        history = read_csv(folder / "training_metrics.csv")
        assert [int(r["epoch"]) for r in history] == list(range(1, 101))
        assert all(int(r["optimizer_updates"]) == 44 * int(r["epoch"]) for r in history)
        assert all(np.isfinite(float(r[k])) for r in history for k in ("train_loss", "val_loss", "learning_rate"))
        total_seconds = sum(float(r["wall_seconds_with_snapshot"]) for r in history)
        timing.append(dict(seed=seed, epochs=100, updates=4400, epoch_callback_seconds=total_seconds,
                           epoch_median_seconds=float(np.median([float(r["seconds"]) for r in history])),
                           fullimage_inference_median_seconds=float(np.median([r["fullimage_inference_seconds"] for r in snapshots])),
                           final_learning_rate=float(history[-1]["learning_rate"]),
                           finished_unix=read_json(folder / "status.json")["finished_unix"]))
        print(f"seed {seed}: 20 hashes, 620 rows, primary/secondary predictions and 100 epochs verified", flush=True)
    for label, folder in (("Spotiflow_Z0", "Z0_default"), ("Spotiflow_Z1", "Z1_validation_calibrated")):
        metrics = read_json(parent / "01_pretrained" / folder / "metrics.json")
        records.append(dict(model=label, seed=0, role="pretrained", epoch=0, threshold=None,
                            **{k: metrics[k] for k in ("tp", "fp", "fn", "precision", "recall", "f1", "weak_tp", "weak_n", "weak_recall")},
                            mean_error_px=None, median_error_px=None, rmse_error_px=None, macro_f1=None, count_mae=None))
    # Persist detailed per-image material only in the local, ignored analysis folder.
    table(out / "per_seed_metrics.csv", records)
    table(out / "per_image_metrics.csv", images)
    table(out / "all_epoch_threshold_recomputed.csv", grid_records)
    table(out / "checkpoint_learning.csv", learning)
    table(out / "training_timing.csv", timing)
    for name, rows in (("per_seed_metrics.csv",records),("checkpoint_learning.csv",learning),("training_timing.csv",timing)):
        table(paper/name,rows)
    dump(paper/"method_lock_cuda_20261030.json",dict(method="Spotiflow general fine-tuned, not trained from scratch",
        protocol=protocol,pretrained_weight_sha256=manifest["files"]["assets/general/best.pt"],
        primary_models=[dict(seed=seed,**read_json(cloud/f"seed_{seed}/selection.json")["primary"],
                             weight_sha256=read_json(cloud/f"seed_{seed}/selection.json")["primary_weight_sha256"])
                        for seed in (42,43,44)],test_evaluated=False,validation_selection_optimism=True))
    summary = []
    metrics_keys = ("f1", "precision", "recall", "tp", "fp", "fn", "weak_recall", "mean_error_px", "macro_f1", "count_mae")
    for model in dict.fromkeys(r["model"] for r in records):
        selected = [r for r in records if r["model"] == model]
        row = dict(model=model, n_training_seeds=0 if selected[0]["seed"] == 0 else len(selected), n_selected_models=len(selected))
        for key in metrics_keys:
            values = [r[key] for r in selected if r[key] is not None]
            row[key + "_mean"] = float(np.mean(values)) if values else None
            row[key + "_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        summary.append(row)
    sizes = []
    for model in ("R0", "G0", "Qglobal", "Spotiflow_F", "Spotiflow_native_secondary"):
        for seed in (42, 43, 44):
            for size in (256, 512):
                rows = [r for r in images if r["model"] == model and r["seed"] == seed and r["size"] == size]
                count = np.sum([[r[k] for k in ("tp", "fp", "fn")] for r in rows], axis=0)
                sizes.append(dict(model=model, seed=seed, size=size, images=len(rows), **{k:float(v) for k,v in counts_metrics(count).items()}))
    comparisons = []
    rng = np.random.default_rng(20261025)
    draws = rng.integers(0, 22, size=(10000, 22))
    def image_counts(model):
        return np.asarray([[[next(r for r in images if r["model"] == model and r["seed"] == seed and r["sample_id"] == sid)[k]
                             for k in ("tp", "fp", "fn")] for sid in data["sample_id"]] for seed in (42,43,44)])
    def scores(model):
        return np.asarray([next(r for r in records if r["model"] == model and r["seed"] == seed)["f1"] for seed in (42,43,44)])
    for baseline in ("G0", "R0"):
        delta = scores("Spotiflow_F") - scores(baseline)
        a, b = image_counts("Spotiflow_F"), image_counts(baseline)
        sampled_a, sampled_b = a[:, draws].sum(2), b[:, draws].sum(2)
        diagnostic = (counts_metrics(sampled_a)["f1"] - counts_metrics(sampled_b)["f1"]).mean(0)
        comparisons.append(dict(comparison="Spotiflow_F-" + baseline, n_training_pairs=3,
                                 mean_delta_f1=float(delta.mean()), sd_delta_f1=float(delta.std(ddof=1)),
                                 delta_seed_42=float(delta[0]), delta_seed_43=float(delta[1]), delta_seed_44=float(delta[2]),
                                 positive_seeds=int((delta>0).sum()), exact_sign_flip_p=exact_seed_test(delta),
                                 holm_p=0., conditional_image_bootstrap_low=float(np.quantile(diagnostic,.025)),
                                 conditional_image_bootstrap_high=float(np.quantile(diagnostic,.975)),
                                 confirmatory=False))
    for row, p in zip(comparisons, holm([r["exact_sign_flip_p"] for r in comparisons])):row["holm_p"] = float(p)
    complement = []
    for seed in (42,43,44):
        a, b = truth_matches[("Spotiflow_F",seed)], truth_matches[("G0",seed)]
        both = only_a = only_b = neither = 0; common_a=[];common_b=[]
        for sid, truth in zip(data["sample_id"],data["y"]):
            sa,sb=set(a[sid]),set(b[sid]);common=sa&sb
            both+=len(common);only_a+=len(sa-sb);only_b+=len(sb-sa);neither+=len(truth)-len(sa|sb)
            common_a.extend(a[sid][i] for i in common);common_b.extend(b[sid][i] for i in common)
        complement.append(dict(seed=seed,both_detected=both,spotiflow_only=only_a,g0_only=only_b,both_missed=neither,
                               common_truth_spotiflow_mean_error_px=float(np.mean(common_a)),
                               common_truth_g0_mean_error_px=float(np.mean(common_b))))
    audit["checks"].update(snapshot_weight_and_candidate_hashes=60, distinct_snapshots_per_seed=20,
                            epoch_threshold_rows_recomputed=1860, training_epochs=300, selected_prediction_sets_rematched=15,
                            selected_threshold_files=6, no_nonfinite_losses=True, common_validation_truth_points=372)
    audit["warnings"] = ["No server SHA256 for the returned tarball; gzip CRC and internal saved hashes pass",
                         "Manifest cuda_verified/cloud_uploaded fields describe packaging time; runtime protocol and smoke are authoritative",
                         "Exactly three formal fine-tuning seeds; no extra search or training in this analysis",
                         "General-pretraining overlap not ruled out; local controls start from scratch",
                         "Different optimizer, augmentation, crop, updates, precision and hardware prevent architectural or speed attribution",
                         "Validation selects checkpoints and thresholds; neither bootstrap nor seed p-values remove selection optimism"]
    audit["analysis_environment"] = dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,torch=torch.__version__)
    audit["source_sha256"] = sha256(HERE)
    if args.probe_deployment:
        probes, parameters = deployment_probe(cloud,bundle,data,root)
        table(out/"deployment_reload_probe.csv",probes)
        table(paper/"deployment_reload_probe.csv",probes)
        audit["checks"]["deployment_reload_fixed_images"] = len(probes)
        audit["model_parameters"] = parameters
    dump(out / "audit.json",audit)
    for name, rows in (("model_summary.csv",summary),("paired_comparisons.csv",comparisons),("size_metrics.csv",sizes),
                       ("error_complementarity.csv",complement)):
        table(out/name,rows)
        table(paper/name,rows)
    dump(paper/"analysis_provenance.json",dict(plan=plan,audit=audit,training_timing=timing))
    make_figures(paper,records,comparisons,learning,cloud,sizes)
    dump(out/"results.json",dict(summary=summary,comparisons=comparisons,complementarity=complement,timing=timing))
    print(json.dumps(dict(summary=summary,comparisons=comparisons,complementarity=complement,timing=timing),indent=2),flush=True)


def make_figures(folder, records, comparisons, learning, cloud, sizes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors={"R0":"#777777","G0":"#0072B2","Qglobal":"#56B4E9","Spotiflow_Z0":"#999999",
            "Spotiflow_Z1":"#E69F00","Spotiflow_F":"#009E73","Spotiflow_native_secondary":"#CC79A7"}
    def values(model,key):return np.asarray([r[key] for r in records if r["model"]==model],dtype=float)
    def export(fig,name):
        fig.savefig(folder/(name+".png"),dpi=240,facecolor="white")
        svg = folder/(name+".svg")
        fig.savefig(svg,facecolor="white")
        svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines())+"\n", encoding="utf-8")
        plt.close(fig)
    with plt.rc_context({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,
                         "axes.spines.right":False,"svg.fonttype":"none","pdf.fonttype":42}):
        models=["R0","G0","Spotiflow_Z0","Spotiflow_Z1","Spotiflow_F"]
        labels=["R0\nn=3","G0\nn=3","Z0 transfer\n1 checkpoint","Z1 calibrated\n1 checkpoint","Spotiflow FT\nn=3"]
        fig,ax=plt.subplots(figsize=(9,5.4),layout="constrained")
        for i,m in enumerate(models):
            v=values(m,"f1");mean=v.mean()
            ax.bar(i,mean,color=colors[m],edgecolor="black",linewidth=.6,hatch="//" if len(v)==1 else None,
                   yerr=v.std(ddof=1) if len(v)>1 else None,capsize=4)
            if len(v)>1:ax.plot(i+np.linspace(-.12,.12,len(v)),v,"o",color="black",ms=4)
            ax.text(i,mean+.026,f"{mean:.4f}",ha="center",fontsize=10)
        ax.set(xticks=range(5),xticklabels=labels,ylim=(0,1),ylabel="Development micro-F1 (3 px)",
               title="Local adaptation makes Spotiflow the strongest evaluated configuration")
        fig.supxlabel("Bars: mean ± sample SD for three training seeds; transfer bars: one checkpoint, no SD.",fontsize=9)
        export(fig,"model_comparison")
        fig,axes=plt.subplots(1,2,figsize=(10,4.8),layout="constrained")
        for j,base in enumerate(("G0","R0")):
            ax=axes[j]
            for i,seed in enumerate((42,43,44)):
                a=next(r for r in records if r["model"]==base and r["seed"]==seed)["f1"]
                b=next(r for r in records if r["model"]=="Spotiflow_F" and r["seed"]==seed)["f1"]
                ax.plot([0,1],[a,b],marker=["o","s","^"][i],label=f"seed {seed}",linewidth=1.5)
            row=comparisons[j]
            ax.set(xticks=[0,1],xticklabels=[base,"Spotiflow FT"],ylim=(.78,.89),ylabel="Development micro-F1",
                   title=f"ΔF1 = +{100*row['mean_delta_f1']:.2f} pp; Holm p={row['holm_p']:.2f} (ns)")
            ax.legend(frameon=False)
        fig.supxlabel("Same seed identifiers; different training recipes. Exact n=3 tests are exploratory.",fontsize=9)
        export(fig,"paired_seed_comparison")
        fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout="constrained")
        for seed,color in zip((42,43,44),("#0072B2","#D55E00","#009E73")):
            rows=read_csv(cloud/f"seed_{seed}/training_metrics.csv")
            axes[0].plot([int(r["epoch"]) for r in rows],[float(r["val_loss"]) for r in rows],color=color,label=f"seed {seed}",alpha=.8)
            rows=[r for r in learning if r["seed"]==seed]
            axes[1].plot([r["epoch"] for r in rows],[r["best_fullimage_f1"] for r in rows],color=color,marker="o",ms=3,label=f"seed {seed}")
            choice=next(r for r in records if r["model"]=="Spotiflow_F" and r["seed"]==seed)
            axes[1].plot(choice["epoch"],choice["f1"],marker="*",ms=14,color=color,markeredgecolor="black",linestyle="none")
        axes[0].set(xlabel="Epoch",ylabel="Native validation loss (256 crop)",title="Native crop loss is noisy")
        axes[1].set(xlabel="Epoch",ylabel="Full-image F1, best registered threshold",ylim=(.75,.9),title="Primary checkpoints selected at epochs 45–60")
        axes[0].legend(frameon=False);axes[1].legend(frameon=False)
        fig.supxlabel("All 20 registered checkpoints shown; stars mark selected epochs, not statistical significance.",fontsize=9)
        export(fig,"training_and_selection")
        fig,axes=plt.subplots(1,3,figsize=(12,4.8),layout="constrained")
        models=["R0","G0","Spotiflow_F"]
        for ax,key,title in zip(axes,("precision","recall","weak_recall"),("Precision","Recall","Weak-point recall (96 reference points)")):
            for i,m in enumerate(models):
                v=values(m,key);ax.bar(i,v.mean(),yerr=v.std(ddof=1),capsize=4,color=colors[m],edgecolor="black",linewidth=.6)
                ax.plot(i+np.linspace(-.1,.1,3),v,"o",color="black",ms=3)
            ax.set(xticks=range(3),xticklabels=["R0","G0","Spotiflow FT"],ylim=(0,1),title=title)
        fig.supxlabel("Same selected checkpoints as the F1 comparison; bars show mean ± sample SD across seeds.",fontsize=9)
        export(fig,"precision_recall_diagnostics")
    dump(folder/"figure_manifest.json",dict(formats=["png","svg"],png_dpi=240,bar_baseline=0,
        uncertainty="sample SD across three training seeds; none for one pretrained checkpoint",
        transformations="micro-F1 per seed from pooled 22-image counts; no smoothing",
        figures={"model_comparison":"Fine-tuned Spotiflow has the highest observed development F1; methods use different recipes.",
                 "paired_seed_comparison":"All three paired seed identifiers favor Spotiflow; exploratory adjusted tests are ns.",
                 "training_and_selection":"All checkpoints retained; native crop loss differs from full-image point-F1 selection.",
                 "precision_recall_diagnostics":"Precision, overall recall and weak-point recall must be considered separately."}))


if __name__ == "__main__":
    if hasattr(sys.stdout,"reconfigure"):sys.stdout.reconfigure(encoding="utf-8")
    main()
