"""Step 1 only: frozen-model CPU inference and export-anchored peak diagnostics.

No fitting, optimizer, architecture change, threshold search or label correction.
Pixel signs: negative dy = up; negative dx = left in the original image.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

import numpy as np
from scipy.special import expit

CODE_ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(CODE_ROOT / "deepblink_pytorch"), str(CODE_ROOT / "deepblink_pytorch/scripts")]
from research_v4.common import dump, rows, read_json, sha256, write_csv
from research_v4.engine import configure_device, make_model, model_hash
from deepblink_torch.cli import predict
from deepblink_torch.preprocessing import prepare_image
from deepblink_torch.round2_protocol import local_peak_mask, threshold_mask
from deepblink_torch.evaluation import match_coordinates

REL_OUTPUT = "docs/next_stage_plan_20261025/baseline_cuda_analysis_20261102/diagnostics"
FEATURES = "outputs/next_stage_v3_closing_controls/02_analysis/validation_truth_features.csv"
PARITY_RADIUS = .005  # px; source attribution only, never the detection tolerance
EDGE_BAND = 10.       # original FOV extent [0,H) x [0,W); sensitivity at 3/32px


def capture_sources(output, shape, dense):
    """Keep source indices before any coordinate cast, floor or FOV filtering."""
    h, w = map(int, shape)
    if not np.isfinite(output).all():
        raise FloatingPointError("Nonfinite frozen-model output")
    cell = 1 if dense else 4
    if dense:
        probability = expit(output[:h, :w, 0])
        rr, cc = np.nonzero(local_peak_mask(probability))
        scores = probability[rr, cc]
    else:
        rr, cc = np.indices(output.shape[:2])
        rr, cc = rr.ravel(), cc.ravel()
        scores = output[rr, cc, 0]
    anchors = np.column_stack([rr, cc]) * cell
    offsets = output[rr, cc, 1:3] * cell
    full_precision = anchors + offsets
    serialized = full_precision.astype(np.float32)
    old_inside = (full_precision >= 0).all(1) & (full_precision < shape).all(1)
    inside = (serialized >= 0).all(1) & (serialized < shape).all(1)
    roundout = old_inside & ~inside
    audit = dict(raw_candidates=len(rr), rounded_out_candidates=int(roundout.sum()),
                 rounded_out_scores=scores[roundout].astype(float).tolist())
    source = dict(coordinates=serialized[inside], anchors=anchors[inside], offsets=offsets[inside],
                  full_precision=full_precision[inside], scores=scores[inside].astype(np.float32),
                  indices=np.column_stack([rr, cc])[inside])
    return source, audit


def select_sources(source, threshold):
    mask = threshold_mask(source["scores"], threshold)
    return {k: v[mask] for k, v in source.items()}


def link_export(source, exported, exported_scores=None):
    """Never silently substitute CPU matches for the accepted GPU prediction set."""
    coords = source["coordinates"]
    score_order = exported_scores is None or (len(source["scores"]) == len(exported_scores)
                    and np.allclose(source["scores"], exported_scores, atol=1e-4, rtol=0))
    if len(coords) == len(exported) and np.allclose(coords, exported, atol=PARITY_RADIUS, rtol=0) and score_order:
        return dict(enumerate(range(len(coords)))), "ordered"
    match = match_coordinates(coords, exported, PARITY_RADIUS)
    mapping = {int(gpu): int(cpu) for cpu, gpu in zip(match.matched_pred_indices, match.matched_true_indices)}
    # Ambiguous near-identical alternatives cannot establish a unique source.
    for gpu, cpu in list(mapping.items()):
        distance = np.linalg.norm(coords - exported[gpu], axis=1)
        if np.count_nonzero(distance <= PARITY_RADIUS) > 1:
            del mapping[gpu]
    return mapping, "unique_coordinate_match"


def edge_distance(point, shape):
    y, x = map(float, point)
    h, w = map(int, shape)
    return min(y, x, h-y, w-x)


def decompose(exported, truth, anchor, offset, raw, serialized, cell):
    truth = np.asarray(truth, dtype=float)
    offset = np.asarray(offset, dtype=float)
    target_anchor = np.floor(truth / cell) * cell
    target_offset = truth - target_anchor
    shift = anchor - target_anchor
    offset_error = offset - target_offset
    cast_error = serialized.astype(float) - raw
    backend_error = exported.astype(float) - serialized.astype(float)
    residual = exported.astype(float) - truth.astype(float)
    np.testing.assert_allclose(shift + offset_error + cast_error + backend_error, residual, atol=1e-10, rtol=0)
    return target_anchor, target_offset, shift, offset_error, cast_error, backend_error


def input_files(data_root):
    base = data_root / "outputs/next_stage_v4_recipe_20261101"
    trials = {}
    inputs = [data_root / FEATURES]
    for method, directory in (("H0", "extracted_h0"), ("R0", "extracted_r0")):
        queue = base / directory / "outputs/research_v4_cuda" / ("baseline_" + method.lower())
        inputs += [queue / "protocol.json", queue / "method_locks.json"]
        for seed in (42, 43, 44):
            folder = queue / (method + "_cuda_reference") / f"seed_{seed}"
            trials[(method, seed)] = folder
            inputs += [folder / name for name in ("best_model.pt", "selection.json", "status.json")]
            inputs += [folder / "primary" / name for name in ("prediction_points.csv", "matches.csv", "per_image_metrics.csv", "metrics.json")]
    h0 = base / "extracted_h0/outputs/research_v4_cuda/baseline_h0"
    inputs += [h0 / name for name in ("boundary_roundout_events.jsonl", "numerical_decode_addendum.json", "epoch60_replay_check.json", "boundary_recovery_complete.json")]
    sealed = base / "04_bundles/ce2d1821ceec/p1_v4_training"
    inputs += [sealed / "assets/development_only.npz", sealed / "protocol_proposed.yaml"]
    return trials, inputs, sealed


def load_validation(sealed):
    import yaml
    spec = yaml.safe_load((sealed / "protocol_proposed.yaml").read_text(encoding="utf-8"))
    path = sealed / "assets/development_only.npz"
    if sha256(path) != spec["data"]["development_sha256"]:
        raise ValueError("Frozen development image archive changed")
    keys = []
    with np.load(path, allow_pickle=True) as archive:
        if set(archive.files) != set(spec["data"]["allowed_npz_keys"]):
            raise ValueError("Unexpected archive/test keys")
        data = {}
        for key in ("x", "y", "sample_id", "original_shape"):
            keys.append(key + "_valid")
            data[key] = archive[key + "_valid"]
    if len(data["x"]) != 22 or sum(map(len, data["y"])) != 372:
        raise ValueError("Validation identity/count differs")
    return {str(sid): dict(image=np.asarray(image, np.float32), truth=np.asarray(truth, np.float32), shape=np.asarray(shape, int))
            for sid, image, truth, shape in zip(data["sample_id"], data["x"], data["y"], data["original_shape"])}, keys


def diagnostic(data_root, out):
    import torch
    trials, inputs, sealed = input_files(data_root)
    immutable = {str(p): sha256(p) for p in inputs}
    training_sources = [CODE_ROOT / "deepblink_pytorch" / relative for relative in
                        ("deepblink_torch/round2_protocol.py", "scripts/research_v4/engine.py", "scripts/research_v4/augmentation.py")]
    immutable.update({str(p): sha256(p) for p in training_sources})
    if (out / "provenance.json").exists():
        raise FileExistsError("Completed diagnostic exists; use --plots-only or a new subdirectory of diagnostics")
    validation, keys = load_validation(sealed)
    features = {(r["sample_id"], int(r["true_index"])): r for r in rows(data_root / FEATURES)}
    z = np.asarray([float(r["local_zscore"]) for r in features.values()])
    if len(features) != 372 or not np.isfinite(z).all():
        raise ValueError("Incomplete local signal/noise proxy")
    cuts = np.quantile(z, [.25, .5, .75])
    cache = {}
    for (method, seed), folder in trials.items():
        choice = read_json(folder / "selection.json")["primary"]
        if sha256(folder / "best_model.pt") != choice["checkpoint_sha256"] or read_json(folder / "status.json")["status"] != "complete":
            raise ValueError("Selected frozen model identity differs")
        predictions = rows(folder / "primary/prediction_points.csv")
        matches = rows(folder / "primary/matches.csv")
        per_image = rows(folder / "primary/per_image_metrics.csv")
        if len(per_image) != 22 or {r["sample_id"] for r in per_image} != set(validation):
            raise ValueError("Image export identities differ")
        for sid, item in validation.items():
            p = [r for r in predictions if r["sample_id"] == sid]
            coords = np.asarray([[float(r["y"]), float(r["x"])] for r in p], np.float32).reshape(-1, 2)
            mm = [r for r in matches if r["sample_id"] == sid]
            image_metrics = next(r for r in per_image if r["sample_id"] == sid)
            rematch = match_coordinates(coords, item["truth"], 3.)
            if (rematch.tp, rematch.fp, rematch.fn) != tuple(int(float(image_metrics[k])) for k in ("tp", "fp", "fn")):
                raise ValueError("Accepted export rematching failed")
            if {(int(r["pred_index"]), int(r["true_index"])) for r in mm} != set(zip(map(int, rematch.matched_pred_indices), map(int, rematch.matched_true_indices))):
                raise ValueError("Accepted match identity differs")
            cache[(method, seed, sid)] = dict(predictions=p, coords=coords, matches=mm)
    configure_device("cpu")
    result, parity, rounding = [], [], []
    for (method, seed), folder in trials.items():
        start = time.perf_counter()
        payload = torch.load(folder / "best_model.pt", map_location="cpu", weights_only=True)
        model = make_model(method, "cpu")
        model.load_state_dict(payload["model_state_dict"])
        model.eval()
        if model_hash(model) != payload["model_state_sha256"]:
            raise ValueError("Model tensor identity differs")
        dense, cell = method == "H0", 1 if method == "H0" else 4
        choice = read_json(folder / "selection.json")["primary"]
        threshold = choice["calibrated_probability"]
        for sid, item in validation.items():
            image, truth, shape = item["image"], item["truth"], item["shape"]
            tensor = torch.from_numpy(prepare_image(image, 512)[None, None]).contiguous(memory_format=torch.channels_last)
            with torch.inference_mode():
                prediction = predict._forward_with_geometry(model, tensor, *map(int, shape))[0].float().numpy()
            source, audit = capture_sources(prediction, shape, dense)
            rounding.append(dict(method=method, seed=seed, sample_id=sid, **audit))
            source = select_sources(source, threshold)
            frozen = cache[(method, seed, sid)]
            mapping, mapping_method = link_export(source, frozen["coords"], np.asarray([float(r["p"]) for r in frozen["predictions"]]))
            matched_errors = [float(np.linalg.norm(frozen["coords"][i] - source["coordinates"][j])) for i, j in mapping.items()]
            parity.append(dict(method=method, seed=seed, sample_id=sid, gpu_predictions=len(frozen["coords"]),
                               cpu_predictions=len(source["coordinates"]), linked=len(mapping), mapping_method=mapping_method,
                               max_coordinate_difference_px=max(matched_errors, default=0.)))
            other = cache[("R0" if dense else "H0", seed, sid)]
            other_truth = {int(r["true_index"]) for r in other["matches"]}
            for match in frozen["matches"]:
                pi, ti = int(match["pred_index"]), int(match["true_index"])
                exported, target = frozen["coords"][pi], truth[ti]
                feature = features[(sid, ti)]
                np.testing.assert_allclose(target, [float(feature["y"]), float(feature["x"])], atol=1e-6, rtol=0)
                local_z = float(feature["local_zscore"])
                edge = edge_distance(target, shape)
                row = dict(method=method, seed=seed, sample_id=sid, true_index=ti, pred_index=pi,
                           height=int(shape[0]), width=int(shape[1]), true_y=float(target[0]), true_x=float(target[1]),
                           pred_y=float(exported[0]), pred_x=float(exported[1]), score=float(frozen["predictions"][pi]["p"]),
                           dy_px=float(exported[0]-target[0]), dx_px=float(exported[1]-target[1]),
                           common_tp=ti in other_truth, edge_distance_px=edge, region="boundary" if edge < EDGE_BAND else "center",
                           snr_proxy_local_zscore=local_z, snr_quartile=int(np.searchsorted(cuts, local_z, side="right")+1),
                           cell_size_px=cell, cpu_source_linked=pi in mapping)
                if pi in mapping:
                    j = mapping[pi]
                    anchor, offset = source["anchors"][j], source["offsets"][j]
                    raw, serialized = source["full_precision"][j], source["coordinates"][j]
                    ta, to, shift, error, cast, backend = decompose(exported, target, anchor, offset, raw, serialized, cell)
                    target_index = (ta / cell).astype(int)
                    oracle_offset = prediction[target_index[0], target_index[1], 1:3] * cell
                    row.update(peak_equals_target_anchor=bool(np.array_equal(anchor, ta)),
                               peak_anchor_occupied_by_other_truth=any(np.array_equal(np.floor(t/cell)*cell, anchor) for k,t in enumerate(truth) if k != ti),
                               offset_at_target_anchor_only_gt_conditioned=True)
                    for axis, index in (("y", 0), ("x", 1)):
                        row.update({"peak_anchor_"+axis: int(anchor[index]), "target_anchor_"+axis: int(ta[index]),
                                    "pred_offset_"+axis+"_px": float(offset[index]), "target_offset_"+axis+"_px": float(to[index]),
                                    "peak_anchor_shift_"+axis+"_px": float(shift[index]), "offset_error_"+axis+"_px": float(error[index]),
                                    "float32_cast_error_"+axis+"_px": float(cast[index]), "gpu_minus_cpu_"+axis+"_px": float(backend[index]),
                                    "offset_error_at_gt_anchor_"+axis+"_px": float(oracle_offset[index]-to[index])})
                result.append(row)
        if model_hash(model) != payload["model_state_sha256"]:
            raise ValueError("Inference changed frozen model state")
        print(json.dumps(dict(method=method, seed=seed, validation_images=22, seconds=round(time.perf_counter()-start, 2))), flush=True)
    for path, digest in immutable.items():
        if sha256(path) != digest:
            raise ValueError("Immutable input/training code changed: " + path)
    # Missing raw-source mappings are retained as blank fields, never discarded.
    fields = list(dict.fromkeys(k for row in result for k in row))
    write_csv(out / "peak_offset_diag.csv", result, fields)
    write_csv(out / "cpu_gpu_parity.csv", parity)
    write_csv(out / "rounding_audit.csv", rounding)
    summary = summarize(result)
    write_csv(out / "peak_offset_summary.csv", summary)
    metadata = dict(stage="step1_only", no_training=True, optimizer_updates=0, device="cpu", torch=str(torch.__version__),
                    test_arrays_read=False, archive_keys_read=keys, snr_proxy="existing local_zscore: (rounded-center intensity - mean of clipped radius10 patch) / patch population SD; not physical shot-noise SNR",
                    snr_quartile_cuts=cuts.tolist(), snr_quantile_reference="372 unique validation truth points; shared across methods/seeds",
                    edge_band_px=EDGE_BAND, edge_definition="min(y,x,H-y,W-x) < 10 on truth coordinates; 3/32px sensitivity in report",
                    detection_matching_radius_px=3., cpu_source_matching_radius_px=PARITY_RADIUS,
                    immutable_inputs=immutable, source_sha256=sha256(__file__), inference_images=132,
                    n_tp_events=len(result), n_unique_truth_points=len(features), n_source_linked=sum(r["cpu_source_linked"] for r in result),
                    hist_edges=np.linspace(-3, 3, 61).tolist(), no_posthoc_coordinate_correction=True,
                    provenance_note="GPU exports define accepted TP identities; CPU inference only attributes source peaks and offsets",
                    steps_2_to_4="not_started_waiting_for_user_confirmation")
    dump(out / "provenance.json", metadata)
    return result, metadata


def summarize(data):
    summary = []
    for method in ("H0", "R0"):
        for seed in ("pooled", 42, 43, 44):
            for population in ("all_tp", "common_tp"):
                for region in ("all", "boundary", "center"):
                    for quartile in ("all", 1, 2, 3, 4):
                        group = [r for r in data if r["method"] == method and (seed == "pooled" or r["seed"] == seed)
                                 and (population == "all_tp" or r["common_tp"]) and (region == "all" or r["region"] == region)
                                 and (quartile == "all" or r["snr_quartile"] == quartile)]
                        for axis in ("y", "x"):
                            values = np.asarray([r["d"+axis+"_px"] for r in group])
                            linked = [r for r in group if r["cpu_source_linked"]]
                            row = dict(method=method, seed=seed, population=population, region=region, snr_quartile=quartile, axis=axis,
                                       n_events=len(group), n_unique_gt=len({(r["sample_id"], r["true_index"]) for r in group}),
                                       n_images=len({r["sample_id"] for r in group}), n_source_linked=len(linked))
                            for key in ("mean_px", "sd_px", "median_px", "q05_px", "q25_px", "q75_px", "q95_px", "negative_fraction"):
                                row[key] = None
                            for key in ("mean_peak_anchor_shift_px", "mean_offset_error_px", "mean_float32_cast_error_px", "max_abs_float32_cast_error_px", "mean_gpu_minus_cpu_px", "peak_target_anchor_agreement_fraction"):
                                row[key] = None
                            if len(group):
                                row.update(mean_px=float(values.mean()), sd_px=float(values.std(ddof=1)) if len(values)>1 else None,
                                           median_px=float(np.median(values)), negative_fraction=float((values<0).mean()))
                                row.update({f"q{q:02d}_px": float(np.quantile(values, q/100)) for q in (5,25,75,95)})
                            if linked:
                                row.update(mean_peak_anchor_shift_px=float(np.mean([r[f"peak_anchor_shift_{axis}_px"] for r in linked])),
                                           mean_offset_error_px=float(np.mean([r[f"offset_error_{axis}_px"] for r in linked])),
                                           mean_float32_cast_error_px=float(np.mean([r[f"float32_cast_error_{axis}_px"] for r in linked])),
                                           max_abs_float32_cast_error_px=float(np.max(np.abs([r[f"float32_cast_error_{axis}_px"] for r in linked]))),
                                           mean_gpu_minus_cpu_px=float(np.mean([r[f"gpu_minus_cpu_{axis}_px"] for r in linked])),
                                           peak_target_anchor_agreement_fraction=float(np.mean([r["peak_equals_target_anchor"] for r in linked])))
                            summary.append(row)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plots-only", action="store_true")
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    allowed = (data_root / REL_OUTPUT).resolve()
    out = args.output.resolve() if args.output else allowed
    if not out.is_relative_to(allowed):
        raise ValueError("All outputs must stay under the requested diagnostics directory")
    out.mkdir(parents=True, exist_ok=True)
    if args.plots_only:
        from plot_peak_offset_diag import load_diag
        data, metadata = load_diag(out)
    else:
        data, metadata = diagnostic(data_root, out)
    from plot_peak_offset_diag import render
    render(out, data, metadata)


if __name__ == "__main__":
    main()
