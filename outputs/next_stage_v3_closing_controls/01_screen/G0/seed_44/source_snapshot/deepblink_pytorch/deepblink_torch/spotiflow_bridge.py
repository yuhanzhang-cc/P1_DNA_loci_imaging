"""Development-only data export and shared point evaluation for Spotiflow."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import tifffile

from .experiment_log import ExperimentLog, sha256, write_json
from .round2_protocol import THRESHOLDS_V3, sweep_candidate_counts, threshold_mask
from .evaluation import match_coordinates, metrics_from_match


ARCHIVE_SHA256 = "5a0083f7fb5d515064a2f27ffbbdffc6ca4cfac5d0de7451913b90a3298d3dae"
SPLIT_SHA256 = "3d3a0b6955f27cc0f2e244872718bdb831607e1f6a04a66a1a1d0e7806c1aed0"


def write_csv(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def checked_coordinates(coords, shape):
    coords = np.asarray(coords, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("Coordinates must be N by 2 in pixel y,x order.")
    if not np.isfinite(coords).all() or np.any(coords < 0) or np.any(coords >= shape):
        raise ValueError("Coordinates are nonfinite or outside the original image.")
    return coords


def export_development(archive_path, split_path, output, *, frozen=True):
    """Export original intensities and pixel labels; never access *_test arrays."""
    hashes = dict(archive_sha256=sha256(archive_path), split_sha256=sha256(split_path))
    if frozen and hashes != dict(archive_sha256=ARCHIVE_SHA256, split_sha256=SPLIT_SHA256):
        raise ValueError("Frozen clean220 inputs changed.")
    with Path(split_path).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_id = {row["sample_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("Duplicate sample IDs in split manifest.")
    config = dict(**hashes, test_arrays_loaded=False, coordinate_order="y,x", units="pixel",
                  intensity="archive values cropped to original shape; no normalization or padding",
                  intended_normalization="Spotiflow native percentile normalization")
    with ExperimentLog(output, config) as log, np.load(archive_path, allow_pickle=True) as archive:
        manifest, seen = [], set()
        for subset, folder in [("train", "train"), ("valid", "val")]:
            images, truths, ids, shapes = [archive[f"{key}_{subset}"]
                                           for key in ("x", "y", "sample_id", "original_shape")]
            if not len(images) == len(truths) == len(ids) == len(shapes):
                raise ValueError("Mismatched archive array lengths.")
            destination = log.output / folder
            destination.mkdir()
            for image, truth, sample, shape in zip(images, truths, ids, shapes):
                sample = str(sample)
                if not sample or Path(sample).name != sample or sample in seen or sample in {"sample_230", "sample_286"}:
                    raise ValueError("Unsafe, duplicated, or excluded sample ID.")
                seen.add(sample)
                row = by_id.get(sample)
                h, w = map(int, shape)
                if row is None or row["subset"] != subset or [h, w] != [int(row["height"]), int(row["width"])]:
                    raise ValueError("Archive geometry/split disagrees with frozen manifest.")
                if image.ndim != 2 or h <= 0 or w <= 0 or h > image.shape[0] or w > image.shape[1]:
                    raise ValueError("Invalid original image geometry.")
                original = image[:h, :w]
                truth = checked_coordinates(truth, (h, w))
                if not np.isfinite(original).all() or len(truth) != int(row["n_points"]):
                    raise ValueError("Invalid image or point count.")
                tif = destination / f"{sample}.tif"
                labels = destination / f"{sample}.csv"
                tifffile.imwrite(tif, original, photometric="minisblack")
                np.savetxt(labels, truth, delimiter=",", header="y,x", comments="", fmt="%.9g")
                if not np.array_equal(tifffile.imread(tif), original):
                    raise ValueError("TIFF export round-trip changed pixels.")
                restored = np.loadtxt(labels, delimiter=",", skiprows=1, ndmin=2).reshape(-1, 2)
                if not np.array_equal(restored.astype(np.float32), truth):
                    raise ValueError("CSV export round-trip changed coordinates.")
                manifest.append(dict(sample_id=sample, subset=subset, height=h, width=w,
                                     n_points=len(truth), image_sha256=sha256(tif), csv_sha256=sha256(labels)))
        expected = {s for s, row in by_id.items() if row["subset"] in ("train", "valid")}
        if seen != expected:
            raise ValueError("Missing or unexpected development samples.")
        if frozen and (sum(r["subset"] == "train" for r in manifest), sum(r["subset"] == "valid" for r in manifest)) != (176, 22):
            raise ValueError("Frozen development split sizes changed.")
        write_csv(log.output / "manifest.csv", manifest, list(manifest[0]))
        log.event("export_verified", images=len(manifest), points=sum(r["n_points"] for r in manifest))
    return manifest


def load_validation(archive_path):
    with np.load(archive_path, allow_pickle=True) as archive:
        return {key: archive[f"{key}_valid"] for key in ("sample_id", "original_shape", "y")}


def validate_candidates(coords, scores, shape):
    coords = checked_coordinates(coords, shape)
    scores = np.asarray(scores, dtype=np.float32)
    if scores.shape != (len(coords),) or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Candidate scores must be one finite probability per point.")
    return coords, scores


def save_candidates(path, candidates, ids, shapes):
    if len(candidates) != len(ids) or len(ids) != len(shapes):
        raise ValueError("Candidate/image count mismatch.")
    checked = [validate_candidates(c, s, shape) for (c, s), shape in zip(candidates, shapes)]
    np.savez_compressed(path, coordinates=np.concatenate([c for c, _ in checked]),
                        scores=np.concatenate([s for _, s in checked]),
                        offsets=np.r_[0, np.cumsum([len(c) for c, _ in checked])],
                        sample_id=np.asarray(ids, dtype=str), original_shape=np.asarray(shapes))


def read_candidates(path, data):
    with np.load(path, allow_pickle=False) as arrays:
        if not np.array_equal(arrays["sample_id"], np.asarray(data["sample_id"], dtype=str)) or not np.array_equal(arrays["original_shape"], data["original_shape"]):
            raise ValueError("Candidate IDs/order/geometry differ from validation data.")
        offsets = arrays["offsets"]
        coords, scores = arrays["coordinates"], arrays["scores"]
        if offsets.dtype.kind not in "iu" or offsets.shape != (len(data["sample_id"]) + 1,) or offsets[0] != 0 or offsets[-1] != len(coords) or np.any(np.diff(offsets) < 0) or len(scores) != len(coords):
            raise ValueError("Invalid candidate offsets.")
        return [validate_candidates(coords[a:b], scores[a:b], shape)
                for a, b, shape in zip(offsets[:-1], offsets[1:], data["original_shape"])]


def evaluate_checkpoints(checkpoints, data):
    """Same 31 thresholds, 3px assignment and selection ordering as v3."""
    if not checkpoints or len(checkpoints) > 20 or len({epoch for epoch, _ in checkpoints}) != len(checkpoints):
        raise ValueError("Require 1..20 distinct checkpoint epochs.")
    rows, bundles = [], {}
    for epoch, candidates in checkpoints:
        if epoch not in {0, *range(5, 101, 5)} or (epoch == 0 and len(checkpoints) != 1):
            raise ValueError("Use epoch 0 only for zero-shot, otherwise registered epochs 5..100.")
        if len(candidates) != len(data["y"]):
            raise ValueError("Missing validation image candidates.")
        counts = np.zeros((31, 3), dtype=np.int64)
        for (coords, scores), truth, shape in zip(candidates, data["y"], data["original_shape"]):
            coords, scores = validate_candidates(coords, scores, shape)
            checked_coordinates(truth, shape)
            counts += np.asarray([r[:3] for r in sweep_candidate_counts(coords, scores, truth)], dtype=np.int64)
        for threshold, (tp, fp, fn) in zip(THRESHOLDS_V3, counts):
            p = tp / (tp + fp) if tp + fp else float(tp + fn == 0)
            r = tp / (tp + fn) if tp + fn else 1.
            f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.
            rows.append(dict(epoch=epoch, threshold=threshold, tp=int(tp), fp=int(fp), fn=int(fn),
                             precision=float(p), recall=float(r), f1=float(f1)))
        bundles[epoch] = candidates
    selected = max(rows, key=lambda row: (row["f1"], row["precision"], -row["epoch"], row["threshold"]))
    per_image = []
    for sample, shape, truth, (coords, scores) in zip(data["sample_id"], data["original_shape"], data["y"], bundles[selected["epoch"]]):
        match = match_coordinates(coords[threshold_mask(scores, selected["threshold"])], truth, 3.)
        metrics = metrics_from_match(match)
        metrics = {k: (None if not np.isfinite(v) else v) for k, v in metrics.items()}
        per_image.append(dict(sample_id=str(sample), height=int(shape[0]), width=int(shape[1]), **metrics))
    return rows, selected, per_image


def spotiflow_candidates(model, image, device):
    """No 0.5 prefilter; preserve native NMS/subpixel flow and probability."""
    points, details = model.predict(image, prob_thresh=0., min_distance=1,
                                    exclude_border=False, subpix=True, peak_mode="fast",
                                    normalizer="auto", verbose=False, device=device)
    coords = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    scores = np.asarray(details.prob, dtype=np.float32).reshape(-1)
    if len(scores) != len(coords) or not np.isfinite(coords).all():
        raise ValueError("Spotiflow returned malformed candidates.")
    inside = (coords >= 0).all(1) & (coords < image.shape).all(1)
    return validate_candidates(coords[inside], scores[inside], image.shape)
