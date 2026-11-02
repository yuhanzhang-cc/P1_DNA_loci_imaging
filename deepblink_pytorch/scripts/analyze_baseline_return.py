"""Offline acceptance of returned R0/H0 and exploratory development comparisons.

Reads the original archives and extracted exports without changing them. No
inference, optimizer updates, new threshold selection or test-array access.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import tarfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "deepblink_pytorch"), str(ROOT / "deepblink_pytorch/scripts")]
from analyze_recipe_return import truth_reference, read_trials, exact_paired, holm
from cloud_recipe_train import check_complete
from research_v4.common import read_json, rows, dump, write_csv, sha256
from deepblink_torch.evaluation import match_coordinates

SEEDS = (42, 43, 44)
CONTRASTS = (("H0_cuda_reference", "R0_cuda_reference"),
             ("H0_cuda_reference", "G0_ref_cuda"), ("G0_ref_cuda", "R0_cuda_reference"))


def verify_archive(archive, extracted):
    expected, name = archive.with_name(archive.name + ".sha256").read_text(encoding="ascii").strip().split(maxsplit=1)
    if name.lstrip("*") != archive.name or sha256(archive) != expected:
        raise ValueError("Archive SHA256/name differs: " + archive.name)
    hashes = {}
    with tarfile.open(archive, "r:gz") as stream:
        for member in stream:
            relative = Path(member.name)
            if relative.is_absolute() or ".." in relative.parts or member.issym() or member.islnk():
                raise ValueError("Unsafe member: " + member.name)
            if not member.isfile():
                continue
            name = relative.as_posix()
            if name in hashes:
                raise ValueError("Duplicate archive member")
            digest = hashlib.sha256()
            with stream.extractfile(member) as source:
                for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            target = extracted / relative
            if not target.resolve().is_relative_to(extracted.resolve()) or sha256(target) != digest.hexdigest():
                raise ValueError("Extracted bytes differ: " + name)
            hashes[name] = digest.hexdigest()
    return hashes


def grid_ap(points):
    """Precision-envelope area at supplied recall knots; no extrapolation to R=1.

    This is explicitly AP31 at the selected epoch, not full ranked detection AP.
    """
    values = sorted((float(p["recall"]), float(p["precision"])) for p in points)
    if not values or any(not (0 <= r <= 1 and 0 <= p <= 1) for r, p in values):
        raise ValueError("Invalid precision/recall grid")
    recall = np.asarray([v[0] for v in values])
    precision = np.asarray([v[1] for v in values])
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    return float(np.sum(np.diff(np.r_[0., recall]) * envelope))


def validate_budget(folder):
    training = rows(folder / "training_metrics.csv")
    sweep = rows(folder / "all_epoch_threshold_metrics.csv")
    snapshots = read_json(folder / "snapshots.json")
    if [int(r["epoch"]) for r in training] != list(range(1, 101)):
        raise ValueError("Epoch sequence differs")
    if [int(r["optimizer_updates"]) for r in training] != list(range(22, 2201, 22)):
        raise ValueError("Optimizer update sequence differs")
    if any(not np.isfinite(float(r[k])) for r in training for k in ("train_loss", "val_loss", "seconds")):
        raise ValueError("Nonfinite training history")
    expected = {(epoch, index) for epoch in range(5, 101, 5) for index in range(31)}
    actual = [(int(r["epoch"]), int(r["threshold_index"])) for r in sweep]
    if len(actual) != 620 or set(actual) != expected or [r["epoch"] for r in snapshots] != list(range(5, 101, 5)):
        raise ValueError("Snapshot/threshold budget differs")
    selection = read_json(folder / "selection.json")
    def key(row):
        return float(row["f1"]), float(row["precision"]), -int(row["epoch"]), int(row["threshold_index"])
    for role in ("primary", "native_secondary"):
        candidates = sweep
        if role == "native_secondary":
            epoch = min(snapshots, key=lambda r: (r["native_val_loss"], r["epoch"]))["epoch"]
            candidates = [r for r in sweep if int(r["epoch"]) == epoch]
        best = max(candidates, key=key)
        chosen = selection[role]
        if any(int(best[k]) != chosen[k] for k in ("epoch", "threshold_index", "tp", "fp", "fn")):
            raise ValueError("Selection does not follow the locked tie-break rule")
    return dict(epochs=100, updates=2200, snapshots=20, threshold_pairs=620,
                selected_epoch=selection["primary"]["epoch"], selection_rules_recomputed=True)


def audit(root, out):
    import torch
    returned = {}
    for archive_name, directory in (("baseline_r0_results", "extracted_r0"),
                                    ("baseline_h0_results", "extracted_h0"),
                                    ("h0_recovery_backup_20261102T224431Z", "extracted_backup")):
        returned[directory] = verify_archive(root / (archive_name + ".tar.gz"), root / directory)
    sealed = root / "04_bundles/ce2d1821ceec/p1_v4_training"
    manifest_sha = sha256(sealed / "bundle_manifest.json")
    for relative, digest in read_json(sealed / "bundle_manifest.json")["files"].items():
        if sha256(sealed / relative) != digest:
            raise ValueError("Original sealed bundle changed")
    queues = {}
    budgets = []
    protocols = []
    for name, directory in (("baseline_r0", "extracted_r0"), ("baseline_h0", "extracted_h0")):
        extracted = root / directory
        if sha256(extracted / "bundle_manifest.json") != manifest_sha:
            raise ValueError("Returned source manifest differs")
        queue = extracted / "outputs/research_v4_cuda" / name
        queues[name] = queue
        check_complete(queue.parent, name)
        protocol = read_json(queue / "protocol.json")
        protocols.append(protocol)
        if protocol["manifest_sha256"] != manifest_sha or protocol["test_arrays_loaded"]:
            raise ValueError("Source/data contract differs")
        if read_json(queue / "controller_status.json")["status"] != "complete":
            raise ValueError("Controller is incomplete")
        for seed in SEEDS:
            folder = next(queue.glob("*/seed_" + str(seed)))
            budgets.append(dict(queue=name, seed=seed, **validate_budget(folder)))
    if any(protocols[0][k] != protocols[1][k] for k in ("gpu", "torch", "device", "manifest_sha256")):
        raise ValueError("R0/H0 backend differs")
    h0 = queues["baseline_h0"]
    addendum = read_json(h0 / "numerical_decode_addendum.json")
    complete = read_json(h0 / "boundary_recovery_complete.json")
    replay = read_json(h0 / "epoch60_replay_check.json")
    if not (complete["completed_results_byte_identical"] and complete["epoch60_tensor_exact"] and replay["exact_tensor_replay"]):
        raise ValueError("Recovery gates did not pass")
    if addendum["training_recipe_changed"] or addendum["test_arrays_loaded"] or addendum["original_manifest_sha256"] != manifest_sha:
        raise ValueError("Unexpected numerical addendum")
    if sha256(h0 / "boundary_overlay_source.py") != addendum["overlay_sha256"]:
        raise ValueError("Overlay source differs from audit")
    backup = root / "extracted_backup/outputs/research_v4_cuda/baseline_h0/H0_cuda_reference/seed_43"
    if sha256(backup / "recovery.pt") != addendum["recovery_sha256"] or sha256(backup / "epoch_060/model.pt") != addendum["failed_model_sha256"]:
        raise ValueError("Pre-recovery backup differs")
    checkpoint = torch.load(backup / "recovery.pt", map_location="cpu", weights_only=True)
    if checkpoint["epoch"] != 59 or checkpoint["updates"] != 1298:
        raise ValueError("Backup is not the committed epoch-59 boundary")
    failed = torch.load(backup / "epoch_060/model.pt", map_location="cpu", weights_only=True)
    tensor_hash = hashlib.sha256()
    for name, tensor in failed["model_state_dict"].items():
        tensor_hash.update(name.encode())
        tensor_hash.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    if len({tensor_hash.hexdigest(), failed["model_state_sha256"], replay["tensor_sha256"], addendum["expected_epoch60_tensor_sha256"]}) != 1:
        raise ValueError("Exact replay tensor hashes disagree")
    protected = dict(final_export_verified=0, pre_recovery_only_verified=0, server_cache_not_returned=0)
    for relative, digest in addendum["protected_files"].items():
        current = [returned[k][relative] for k in ("extracted_r0", "extracted_h0") if relative in returned[k]]
        previous = returned["extracted_backup"].get(relative)
        if any(h != digest for h in current) or previous is not None and previous != digest:
            raise ValueError("Protected file changed: " + relative)
        kind = "final_export_verified" if current else "pre_recovery_only_verified" if previous else "server_cache_not_returned"
        protected[kind] += 1
    events = [json.loads(line) for line in (h0 / "boundary_roundout_events.jsonl").read_text().splitlines() if line.strip()]
    result = dict(passed=True, archived_files={k: len(v) for k, v in returned.items()}, budgets=budgets,
                  backend={k: protocols[0][k] for k in ("gpu", "torch", "device")},
                  exact_replay_tensor_sha256=tensor_hash.hexdigest(), recovery_epoch=59,
                  recovery_updates=1298, protected_files=protected,
                  protection_scope="All returned protected files checked; omitted epoch caches rely on recorded server gate",
                  replay_scope="Original failed tensors independently hashed; replayed epoch60 tensor file omitted from final archive",
                  boundary_event_count=len(events), removed_candidates=sum(e["removed"] for e in events),
                  maximum_removed_score=max((max(e["scores"]) for e in events), default=None),
                  stale_failure_txt_retained=(h0 / "failure.txt").exists(), test_arrays_read=False)
    dump(out / "acceptance.json", result)
    return queues, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    root = data_root / "outputs/next_stage_v4_recipe_20261101"
    out = root / "16_baseline_analysis_20261102"
    out.mkdir(exist_ok=True)
    queues, accepted = audit(root, out)
    valid, truth, weak, _ = truth_reference(data_root)
    queues["recipe"] = root / "08_returned_gpu/extracted/outputs/research_v4_cuda/recipe"
    check_complete(queues["recipe"].parent, "recipe")
    for queue in queues.values():
        if any(read_json(queue / "protocol.json")[k] != accepted["backend"][k] for k in ("gpu", "torch", "device")):
            raise ValueError("G0/R0/H0 comparison backend differs")
    for seed in SEEDS:
        initializations = [read_json(queues[q] / m / f"seed_{seed}/initialization.json")
                           for q, m in (("baseline_r0", "R0_cuda_reference"),
                                        ("baseline_h0", "H0_cuda_reference"), ("recipe", "G0_ref_cuda"))]
        if len({r["sampler_orders_sha256"] for r in initializations}) != 1:
            raise ValueError("Paired sampler order differs")
    records, images, points, radii = [], [], {}, []
    folders = {}
    for queue_name, method in (("baseline_r0", "R0_cuda_reference"), ("baseline_h0", "H0_cuda_reference"), ("recipe", "G0_ref_cuda")):
        ctx = dict(data=data_root, queue=queues[queue_name], out=out)
        result = read_trials(ctx, valid, truth, weak, methods=(method,))
        records.extend(result[0]); images.extend(result[1]); points.update(result[2]); radii.extend(result[3])
        for seed in SEEDS:
            folder = queues[queue_name] / method / f"seed_{seed}"
            folders[(method, seed)] = folder
            sweep = rows(folder / "all_epoch_threshold_metrics.csv")
            for record in records:
                if (record["method"], record["seed"]) == (method, seed):
                    record["ap31_grid_3px"] = grid_ap([r for r in sweep if int(r["epoch"]) == record["epoch"]])
    write_csv(out / "per_seed_metrics.csv", records)
    write_csv(out / "per_image_metrics.csv", images)
    write_csv(out / "radius_sensitivity.csv", radii)
    metrics = ("f1", "precision", "recall", "ap31_grid_3px", "weak_recall", "bright_recall", "macro_f1", "count_mae", "mean_error_px", "rmse_error_px", "tp", "fp", "fn")
    summary = []
    for method in dict.fromkeys(r["method"] for r in records):
        for role in ("primary", "native_secondary"):
            selected = [r for r in records if r["method"] == method and r["role"] == role]
            parameters = {read_json(folders[(method, seed)] / "initialization.json")["parameters"] for seed in SEEDS}
            if len(parameters) != 1:
                raise ValueError("Architecture size changed between seeds")
            row = dict(method=method, role=role, n_seeds=len(selected), parameters=parameters.pop())
            for metric in metrics:
                values = np.asarray([r[metric] for r in selected], float)
                if len(values) != 3 or not np.isfinite(values).all():
                    raise ValueError("Missing or nonfinite per-seed result")
                row.update({metric + "_mean": float(values.mean()), metric + "_sd": float(values.std(ddof=1))})
            summary.append(row)
    write_csv(out / "model_summary.csv", summary)
    analyze_errors(out, records, images, points, radii, folders, valid, truth, weak)
    dump(out / "provenance.json", dict(source_sha256=sha256(__file__), source_revision="working tree; see source hashes",
         reader_sha256=sha256(Path(__file__).with_name("analyze_recipe_return.py")),
         archive_sha256={p.name: sha256(p) for p in root.glob("*.tar.gz")},
         estimand="mean of three per-training-seed 3px micro-F1 values on the same 22 development images / 372 points",
         n_weak=96, tests="two-sided paired seed sign-flips; exploratory family of three contrasts, Holm",
         bootstrap="10000 shared image resamples across all seeds, fixed selected models; not biological or selection-adjusted uncertainty",
         standard_ranked_ap_available=False, ap31_definition="precision-envelope area over the existing 31 threshold policies at the fixed selected epoch",
         sampler_orders_matched=True, initial_weights_matched_across_architectures=False,
         no_new_threshold_selection=True, no_model_inference=True, test_arrays_read=False))
    print(json.dumps(dict(acceptance=accepted, primary_summary=[r for r in summary if r["role"] == "primary"]), indent=2))


def analyze_errors(out, records, images, points, radii, folders, valid, truth, weak):
    primary = {(r["method"], r["seed"]): r for r in records if r["role"] == "primary"}
    comparisons = []
    for a, b in CONTRASTS:
        delta = np.asarray([primary[(a, s)]["f1"] - primary[(b, s)]["f1"] for s in SEEDS])
        counts = []
        for method in (a, b):
            counts.append(np.asarray([[[int(float(r[k])) for k in ("tp", "fp", "fn")]
                         for sid in sorted(valid) for r in images if r["method"] == method and r["seed"] == seed and r["role"] == "primary" and r["sample_id"] == sid]
                         for seed in SEEDS]))
        rng = np.random.default_rng(20261102)
        draws = rng.integers(0, 22, size=(10000, 22))
        def micro(c):
            c = c[:, draws].sum(2)
            return 2 * c[..., 0] / (2 * c[..., 0] + c[..., 1] + c[..., 2])
        boot = (micro(counts[0]) - micro(counts[1])).mean(0)
        shared = []
        for seed in SEEDS:
            aa, bb = points[(a, seed, "primary")], points[(b, seed, "primary")]
            common = sorted(set(aa) & set(bb))
            shared.append(dict(comparison=a + "-" + b, seed=seed, shared_true_points=len(common),
                               a_mean_error_px=float(np.mean([aa[k] for k in common])),
                               b_mean_error_px=float(np.mean([bb[k] for k in common]))))
        comparisons.append(dict(comparison=a + "-" + b, mean_delta_f1=float(delta.mean()),
             seed42_delta=float(delta[0]), seed43_delta=float(delta[1]), seed44_delta=float(delta[2]),
             p_exact=exact_paired(delta), bootstrap_low=float(np.quantile(boot, .025)), bootstrap_high=float(np.quantile(boot, .975))))
        write_csv(out / ("shared_tp_" + a + "_vs_" + b + ".csv"), shared)
    for row, adjusted in zip(comparisons, holm([r["p_exact"] for r in comparisons])):
        row["p_holm"] = adjusted
    write_csv(out / "paired_comparisons.csv", comparisons)
    fp_rows, residuals = [], []
    strata = []
    for (method, seed), folder in folders.items():
        predictions = rows(folder / "primary/prediction_points.csv")
        for sid in sorted(valid):
            selected = [r for r in predictions if r["sample_id"] == sid]
            coords = np.asarray([[float(r["y"]), float(r["x"])] for r in selected], np.float32).reshape(-1, 2)
            match = match_coordinates(coords, truth[sid], 3.)
            shape = [int(valid[sid][k]) for k in ("height", "width")]
            for i in match.unmatched_pred_indices:
                distance = np.linalg.norm(truth[sid] - coords[i], axis=1)
                nearest = int(np.argmin(distance))
                d = float(distance[nearest])
                y, x = map(float, coords[i])
                fp_rows.append(dict(method=method, seed=seed, sample_id=sid, pred_index=int(i),
                    score=float(selected[i]["p"]), nearest_truth_distance_px=d,
                    proxy="duplicate_or_assignment_conflict" if d <= 3 else "near_reference_3to6px" if d <= 6 else "far_from_reference_gt6px",
                    within3px_edge=min(y, x, shape[0] - y, shape[1] - x) < 3, human_cause="pending_review"))
            for pi, ti, distance in zip(match.matched_pred_indices, match.matched_true_indices, match.matched_distances):
                delta = coords[pi] - truth[sid][ti]
                residuals.append(dict(method=method, seed=seed, sample_id=sid, true_index=int(ti),
                     dy_px=float(delta[0]), dx_px=float(delta[1]), distance_px=float(distance)))
        for size in (256, 512):
            subset = [r for r in images if r["method"] == method and r["seed"] == seed and r["role"] == "primary" and int(valid[r["sample_id"]]["height"]) == size]
            tp, fp, fn = [sum(int(float(r[k])) for r in subset) for k in ("tp", "fp", "fn")]
            strata.append(dict(method=method, seed=seed, size=size, images=len(subset), tp=tp, fp=fp, fn=fn, f1=2 * tp / (2 * tp + fp + fn)))
    write_csv(out / "fp_geometry_proxies.csv", fp_rows)
    write_csv(out / "matched_point_residuals.csv", residuals)
    write_csv(out / "size_strata.csv", strata)
    summaries = []
    for method in dict.fromkeys(r["method"] for r in fp_rows):
        for seed in SEEDS:
            selected = [r for r in fp_rows if r["method"] == method and r["seed"] == seed]
            row = dict(method=method, seed=seed, fp=len(selected), edge_fp=sum(r["within3px_edge"] for r in selected))
            for category in ("duplicate_or_assignment_conflict", "near_reference_3to6px", "far_from_reference_gt6px"):
                row[category] = sum(r["proxy"] == category for r in selected)
            if row["fp"] != primary[(method, seed)]["fp"]:
                raise ValueError("FP decomposition does not reconcile")
            summaries.append(row)
    write_csv(out / "fp_geometry_summary.csv", summaries)


if __name__ == "__main__":
    main()
