"""Bounded, development-only v3 experiment. Never materializes test arrays."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from deepblink_torch.cli import predict
from deepblink_torch.data_protocol import get_prediction_matrix
from deepblink_torch.losses import combined_dice_rmse_loss
from deepblink_torch.preprocessing import prepare_image
from deepblink_torch.round2_protocol import (
    THRESHOLDS_V3, decode_candidates, dense_heatmap_loss, encode_dense_targets,
    sweep_candidate_counts, threshold_mask, training_orders)
from deepblink_torch.evaluation import match_coordinates, metrics_from_match
from error_context_diagnostics import _extract_patch_features


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/next_stage_v3"
ARCHIVE = ROOT / "outputs/261024_clean220_random/dna_clean220_random.npz"
PLAN = ROOT / "docs/next_stage_plan_20261025/round2_20261026/protocol_v3_proposed.yaml"
PROTOCOL = "pooled220_methods_v3"
NAMES = dict(R0="deepblink", H0="deeploci_heatmap", R1="deeploci_detail",
             S1="deeploci_detail_simple", R2="deeploci_adaptive_context", S2="deeploci_fixed_context")
EPOCHS = list(range(5, 101, 5))
WEAK_CUTOFF = 2.4566830772038957


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source_hashes():
    paths = sorted((ROOT / "deepblink_pytorch/deepblink_torch").rglob("*.py"))
    paths += [Path(__file__), Path(__file__).with_name("training_wake_guard.py"),
              Path(__file__).with_name("error_context_diagnostics.py")]
    paths += sorted((ROOT / "deepblink_pytorch/tests").glob("test_*.py"))
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in paths}


def lock_protocol():
    planned_path = OUT / "00_protocol/planned_spec.json"
    planned = json.loads(planned_path.read_text(encoding="utf-8"))
    if planned["yaml_sha256"] != sha256(PLAN):
        raise ValueError("Proposed protocol changed since its explicit JSON resolution.")
    lock = dict(protocol_id=PROTOCOL, planned_spec_sha256=sha256(planned_path),
                thresholds=THRESHOLDS_V3, checkpoint_epochs=EPOCHS,
                checkpoint_threshold_pairs_per_run=620, matching="maximum_cardinality_then_minimum_distance",
                source_base_commit="9edd52928ea75380c0008fa2ff941ea490a7bd6d",
                split_manifest_sha256=sha256(ROOT / "outputs/261024_clean220_random/frozen_split.csv"),
                archive_sha256=sha256(ARCHIVE), test_evaluated=False,
                max_new_training_tasks=12, screening=["R0", "H0", "R1", "S1"],
                conditional_screening=["R2", "S2"], final_seeds=[42, 43, 44],
                selection="point_micro_F1_then_precision_then_earlier_epoch_then_higher_threshold",
                native_secondary="lowest_native_loss_among_identical_20_snapshots_then_same_threshold_grid",
                human_blinded_annotation_review="pending", biological_independence="unverified",
                validation_selection_optimism=True, user_execution_authorization="2026-10-26 request to begin next optimization")
    path = OUT / "00_protocol/lock.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != lock:
        raise ValueError("Frozen v3 protocol differs.")
    write_json(path, lock)
    return lock


class DevelopmentCorpus:
    """Load each allowed split once and normalize in place to bound host RAM."""
    def __init__(self):
        self.splits = {}
        with np.load(ARCHIVE, allow_pickle=True) as archive:
            for subset in ("train", "valid"):
                data = {k: archive[f"{k}_{subset}"] for k in ("x", "y", "sample_id", "original_shape")}
                data["x"] = data["x"].astype(np.float32, copy=False)
                data["weak"] = []
                for index, (shape, truth) in enumerate(zip(data["original_shape"], data["y"])):
                    h, w = map(int, shape)
                    original = data["x"][index, :h, :w]
                    data["weak"].append(np.array([
                        _extract_patch_features(original, y, x, 10)["local_zscore"] < WEAK_CUTOFF
                        for y, x in truth], dtype=bool))
                    data["x"][index] = prepare_image(original, 512)
                self.splits[subset] = data
        if len(self.splits["train"]["x"]) != 176 or len(self.splits["valid"]["x"]) != 22:
            raise ValueError("Unexpected development split sizes.")
        if sum(map(np.count_nonzero, self.splits["valid"]["weak"])) != 96:
            raise ValueError("Frozen weak-truth definition drifted.")


class DevelopmentDataset(Dataset):
    def __init__(self, data, dense=False):
        self.data, self.dense = data, dense
        self.targets = None
        if not dense:
            targets = []
            for coords, (h, w) in zip(data["y"], data["original_shape"]):
                target = get_prediction_matrix(coords, 512, 4).astype(np.float32)
                target[int(h)//4:, :, 0] = -1
                target[:, int(w)//4:, 0] = -1
                targets.append(target)
            self.targets = np.stack(targets)

    def __len__(self):
        return len(self.data["x"])

    def __getitem__(self, index):
        target = (encode_dense_targets(self.data["y"][index], self.data["original_shape"][index])
                  if self.dense else self.targets[index])
        return torch.from_numpy(self.data["x"][index, None]), torch.from_numpy(target)


def reset_rng(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.xpu.is_available():
        torch.xpu.manual_seed_all(seed)


def make_model(key, device):
    return predict._build_model_for_checkpoint(NAMES[key], cell_size=1 if key == "H0" else 4,
                                                device=torch.device(device))


def forward(model, images, targets):
    if getattr(model, "requires_valid_mask", False):
        mask = torch.nn.functional.interpolate((targets[..., 0] >= 0).float().unsqueeze(1),
                                               size=images.shape[-2:], mode="nearest")
        return model(images, valid_mask=mask)
    return model(images)


def loader(dataset, seed, order=None):
    return DataLoader(dataset, batch_size=8, sampler=order, shuffle=False, num_workers=0,
                      generator=torch.Generator(device="cpu").manual_seed(seed + 100000))


def native_epoch(model, batches, loss_fn, device, optimizer=None):
    model.train(optimizer is not None)
    losses = []
    with torch.set_grad_enabled(optimizer is not None):
        for images, targets in batches:
            images = images.to(device).contiguous(memory_format=torch.channels_last)
            targets = targets.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "xpu"):
                predictions = forward(model, images, targets)
            loss = loss_fn(targets, predictions.float())
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite native loss.")
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            losses.append(float(loss.detach()))
    return float(np.mean(losses))


@torch.no_grad()
def infer_candidates(model, data, device):
    model.eval()
    results = []
    dense = getattr(model, "output_representation", "grid") == "dense_heatmap_offset_logits"
    for image, shape in zip(data["x"], data["original_shape"]):
        tensor = torch.from_numpy(image[None, None]).to(device).contiguous(memory_format=torch.channels_last)
        output = predict._forward_with_geometry(model, tensor, int(shape[0]), int(shape[1]))
        results.append(decode_candidates(output[0].float().cpu().numpy(), shape, dense=dense))
    return results


def save_candidates(path, candidates, data):
    offsets = np.r_[0, np.cumsum([len(c[0]) for c in candidates])]
    np.savez_compressed(path, coordinates=np.concatenate([c[0] for c in candidates]),
                        scores=np.concatenate([c[1] for c in candidates]), offsets=offsets,
                        sample_id=data["sample_id"], original_shape=data["original_shape"])


def load_candidates(path):
    with np.load(path, allow_pickle=False) as z:
        return [(z["coordinates"][a:b], z["scores"][a:b]) for a, b in zip(z["offsets"][:-1], z["offsets"][1:])]


def score_candidates(candidates, data):
    counts = [sweep_candidate_counts(coords, scores, truth) for (coords, scores), truth in zip(candidates, data["y"])]
    rows = []
    weak_n = int(sum(map(np.count_nonzero, data["weak"])))
    for i, threshold in enumerate(THRESHOLDS_V3):
        tp, fp, fn = [sum(c[i][j] for c in counts) for j in range(3)]
        weak_tp = int(sum(np.count_nonzero(weak[c[i][3]]) for weak, c in zip(data["weak"], counts)))
        per_f1 = [2*c[i][0]/(2*c[i][0]+c[i][1]+c[i][2]) if sum(c[i][:3]) else 1. for c in counts]
        rows.append(dict(threshold=threshold, tp=tp, fp=fp, fn=fn,
                         precision=tp/(tp+fp) if tp+fp else 0., recall=tp/(tp+fn),
                         f1=2*tp/(2*tp+fp+fn), macro_f1=float(np.mean(per_f1)),
                         count_mae=float(np.mean([abs(c[i][1]-c[i][2]) for c in counts])),
                         weak_tp=weak_tp, weak_n=weak_n, weak_recall=weak_tp/weak_n))
    return rows


def selection_key(row):
    return row["f1"], row["precision"], -row["epoch"], row["threshold"]


def checkpoint_payload(model, optimizer, key, seed, epoch, val_loss):
    return dict(model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                model_name=NAMES[key], protocol_id=PROTOCOL, cell_size=1 if key == "H0" else 4,
                input_size=512, preprocessing="normalize_original_then_zero_pad_right_bottom",
                output_representation=getattr(model, "output_representation", "grid"),
                parameter_count=sum(p.numel() for p in model.parameters()), seed=seed, epoch=epoch,
                best_val_loss=val_loss, selection_snapshot=True,
                architecture_config=dict(model_name=NAMES[key], round2_key=key))


def save_torch(path, payload):
    temp = path.with_name(path.name + ".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)


def export_selected(destination, candidates, data, selected):
    point_rows, image_rows, match_rows, errors = [], [], [], []
    weak_matched, far_fp = 0, 0
    for sid, shape, truth, weak, (coords, scores) in zip(data["sample_id"], data["original_shape"], data["y"], data["weak"], candidates):
        keep = threshold_mask(scores, selected["threshold"])
        coords, scores = coords[keep], scores[keep]
        match = match_coordinates(coords, truth, 3.)
        weak_matched += int(np.count_nonzero(weak[match.matched_true_indices]))
        false_coords = coords[match.unmatched_pred_indices]
        if len(false_coords):
            nearest = np.linalg.norm(false_coords[:, None]-truth[None], axis=2).min(1) if len(truth) else np.full(len(false_coords), np.inf)
            far_fp += int(np.count_nonzero(nearest > 6))
        image_rows.append(dict(sample_id=str(sid), height=int(shape[0]), width=int(shape[1]),
                               n_true=len(truth), n_pred=len(coords), **{k: (None if not np.isfinite(v) else float(v)) for k, v in metrics_from_match(match).items()}))
        for (y, x), score in zip(coords, scores):
            point_rows.append(dict(sample_id=str(sid), y=float(y), x=float(x), p=float(score)))
        for pi, ti, distance in zip(match.matched_pred_indices, match.matched_true_indices, match.matched_distances):
            match_rows.append(dict(sample_id=str(sid), pred_index=int(pi), true_index=int(ti),
                                   distance_px=float(distance), weak_truth=bool(weak[ti])))
            errors.append(float(distance))
    write_csv(destination / "prediction_points.csv", point_rows)
    write_csv(destination / "per_image_metrics.csv", image_rows)
    write_csv(destination / "matches.csv", match_rows)
    if sum(int(row["tp"]) for row in image_rows) != selected["tp"]:
        raise AssertionError("Accelerated sweep differs from full matcher.")
    if "weak_tp" in selected and weak_matched != selected["weak_tp"]:
        raise AssertionError("Ambiguous match ties changed weak-point accounting; audit before conclusions.")
    return dict(tp_median_error_px=float(np.median(errors)) if errors else None,
                tp_mean_error_px=float(np.mean(errors)) if errors else None,
                tp_rmse_error_px=float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
                fp_farther6_from_truth=far_fp)


def finalize_trial(destination, data):
    rows = list(csv.DictReader((destination / "checkpoint_metrics.csv").open(encoding="utf-8")))
    sweeps = []
    for row in rows:
        epoch = int(row["epoch"])
        for item in json.loads((destination / f"epoch_{epoch:03d}/threshold_metrics.json").read_text(encoding="utf-8")):
            sweeps.append(dict(epoch=epoch, native_val_loss=float(row["val_loss"]), **item))
    selected = max(sweeps, key=selection_key)
    native_epoch_number = min(rows, key=lambda r: (float(r["val_loss"]), int(r["epoch"])))
    native_epoch_number = int(native_epoch_number["epoch"])
    native_selected = max((r for r in sweeps if r["epoch"] == native_epoch_number), key=selection_key)
    source = destination / f"epoch_{selected['epoch']:03d}"
    shutil.copy2(source / "model.pt", destination / "best_model.pt")
    shutil.copy2(destination / f"epoch_{native_epoch_number:03d}/model.pt", destination / "native_selected_model.pt")
    candidates = load_candidates(source / "candidates.npz")
    localization = export_selected(destination, candidates, data, selected)
    write_csv(destination / "all_epoch_threshold_metrics.csv", sweeps)
    result = dict(protocol_id=PROTOCOL, primary=selected, native_secondary=native_selected,
                  selection_delta_f1=selected["f1"]-native_selected["f1"], checkpoint_sha256=sha256(destination / "best_model.pt"),
                  native_checkpoint_sha256=sha256(destination / "native_selected_model.pt"),
                  checkpoint_candidates=20, epoch_threshold_pairs=len(sweeps),
                  test_evaluated=False, independent_validation=False, **localization)
    write_json(destination / "selection.json", result)
    plot_training(destination, selected, native_selected)
    return result


def plot_training(destination, selected, native_selected):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with (destination / "training_metrics.csv").open(encoding="utf-8") as handle:
        history = list(csv.DictReader(handle))
    with (destination / "checkpoint_metrics.csv").open(encoding="utf-8") as handle:
        snapshots = list(csv.DictReader(handle))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = [int(r["epoch"]) for r in history]
    for key, label in [("train_loss", "Train"), ("val_loss", "Validation")]:
        axes[0].plot(epochs, [float(r[key]) for r in history], label=label)
    axes[0].set(xlabel="Epoch", ylabel="Native objective", title="Native training objective")
    axes[0].legend()
    axes[1].plot([int(r["epoch"]) for r in snapshots], [float(r["f1"]) for r in snapshots], marker=".")
    axes[1].scatter([selected["epoch"]], [selected["f1"]], marker="*", s=90, label="Primary selection")
    axes[1].scatter([native_selected["epoch"]], [native_selected["f1"]], marker="x", s=50, label="Native-loss selection")
    axes[1].set(xlabel="Epoch", ylabel="Validation point F1", ylim=(0, 1), title="Registered threshold calibration; validation only")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(destination / "training_curves.png", dpi=150)
    plt.close(fig)


def snapshot(destination, hashes):
    for relative in hashes:
        target = destination / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    write_json(destination / "source_hashes.json", hashes)
    write_json(destination / "environment.json", dict(python=sys.version, torch=str(torch.__version__),
                 device=torch.xpu.get_device_name(0) if torch.xpu.is_available() else "cpu",
                 git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()))


def run_trial(key, seed, corpus, device="xpu", resume=False):
    lock = lock_protocol()
    folder = OUT / "01_screen" / key / f"seed_{seed}"
    hashes = source_hashes()
    config = dict(protocol_id=PROTOCOL, lock_sha256=sha256(OUT / "00_protocol/lock.json"), key=key, model=NAMES[key],
                  seed=seed, epochs=100, batch_size=8, lr=.001, weight_decay=0., device=device,
                  training_amp="bfloat16" if device == "xpu" else "disabled", threads=4, channels_last=True,
                  train_images=176, validation_images=22, updates=2200, source_hashes=hashes,
                  loss="soft_dice_bce_plus_2_anchor_rmse" if key == "H0" else "dice_plus_2_rmse",
                  test_arrays_loaded=False, sampler="independent_precomputed_torch_generator")
    config_path, status_path = folder / "registered_config.json", folder / "status.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError(f"Configuration/source changed: {folder}")
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status["status"] == "complete":
            print(f"Already complete {key} seed {seed}", flush=True)
            return
        if not resume:
            raise RuntimeError(f"Existing partial run needs explicit --resume after audit: {folder}")
    elif resume:
        raise ValueError("No partial run exists to resume.")
    if not config_path.exists():
        configs = list((OUT / "01_screen").glob("*/seed_*/registered_config.json"))
        if len(configs) >= lock["max_new_training_tasks"]:
            raise ValueError("The twelve-task budget is exhausted.")
        snapshot(folder, hashes)
        write_json(config_path, config)
    orders = training_orders(176, 100, seed)
    order_rows = [dict(epoch=e+1, position=i, sample_id=str(corpus.splits["train"]["sample_id"][index]))
                  for e, order in enumerate(orders) for i, index in enumerate(order)]
    write_csv(folder / "sample_order.csv", order_rows)
    reset_rng(seed)
    model = make_model(key, device).to(memory_format=torch.channels_last)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=0.)
    dense = key == "H0"
    train_data = DevelopmentDataset(corpus.splits["train"], dense)
    valid_data = DevelopmentDataset(corpus.splits["valid"], dense)
    loss_fn = dense_heatmap_loss if dense else combined_dice_rmse_loss
    history, checkpoints, first_epoch = [], [], 1
    if resume:
        saved = torch.load(folder / "resume.pt", map_location=device, weights_only=False)
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        history, checkpoints = saved["history"], saved["checkpoints"]
        first_epoch = saved["epoch"] + 1
        torch.set_rng_state(saved["cpu_rng"].cpu())
        if device == "xpu":
            torch.xpu.set_rng_state(saved["xpu_rng"].cpu())
        write_json(folder / f"recovery_epoch_{first_epoch:03d}.json", dict(resumed_at=time.time(), completed_epoch=first_epoch-1, same_configuration=True))
    started = time.time()
    write_json(status_path, dict(status="running", key=key, seed=seed, pid=os.getpid(), first_epoch=first_epoch, started=started))
    try:
        for epoch in range(first_epoch, 101):
            tick = time.perf_counter()
            train_loss = native_epoch(model, loader(train_data, seed, orders[epoch-1]), loss_fn, device, optimizer)
            val_loss = native_epoch(model, loader(valid_data, seed), loss_fn, device)
            row = dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss,
                       seconds=time.perf_counter()-tick, optimizer_updates=epoch*22)
            history.append(row)
            if epoch in EPOCHS:
                epoch_folder = folder / f"epoch_{epoch:03d}"
                epoch_folder.mkdir(parents=True, exist_ok=True)
                save_torch(epoch_folder / "model.pt", checkpoint_payload(model, optimizer, key, seed, epoch, val_loss))
                candidates = infer_candidates(model, corpus.splits["valid"], device)
                save_candidates(epoch_folder / "candidates.npz", candidates, corpus.splits["valid"])
                sweep = score_candidates(candidates, corpus.splits["valid"])
                write_json(epoch_folder / "threshold_metrics.json", sweep)
                write_csv(epoch_folder / "threshold_metrics.csv", sweep)
                best = max(sweep, key=lambda r: (r["f1"], r["precision"], r["threshold"]))
                checkpoints.append(dict(epoch=epoch, val_loss=val_loss, **best))
                write_csv(folder / "checkpoint_metrics.csv", checkpoints)
                print(f"CHECKPOINT {key} seed={seed} epoch={epoch} F1={best['f1']:.6f} P={best['precision']:.4f} R={best['recall']:.4f} weak={best['weak_recall']:.4f} p={best['threshold']:.7f}", flush=True)
            write_csv(folder / "training_metrics.csv", history)
            recovery = checkpoint_payload(model, optimizer, key, seed, epoch, val_loss)
            recovery.update(history=history, checkpoints=checkpoints, cpu_rng=torch.get_rng_state(),
                            xpu_rng=torch.xpu.get_rng_state() if device == "xpu" else None)
            save_torch(folder / "resume.pt", recovery)
            write_json(folder / "progress.json", dict(key=key, seed=seed, **row,
                       checkpoints=len(checkpoints), best_validation_f1=max([r["f1"] for r in checkpoints], default=None)))
            print(f"EPOCH {key} seed={seed} {epoch}/100 loss={train_loss:.6f} val={val_loss:.6f} seconds={row['seconds']:.2f}", flush=True)
        result = finalize_trial(folder, corpus.splits["valid"])
        write_json(status_path, dict(status="complete", key=key, seed=seed, epochs=100,
                   elapsed_seconds=time.time()-started, validation_f1=result["primary"]["f1"], test_evaluated=False))
        print(f"COMPLETE {key} seed={seed}: {json.dumps(result)}", flush=True)
    except BaseException as error:
        write_json(status_path, dict(status="failed_or_interrupted", key=key, seed=seed, error=repr(error), test_evaluated=False))
        raise
    finally:
        del model, optimizer, train_data, valid_data
        if device == "xpu":
            torch.xpu.empty_cache()


def audit(corpus, device):
    lock_protocol()
    rows = list(csv.DictReader((ROOT / "outputs/261024_clean220_random/frozen_split.csv").open(encoding="utf-8")))
    checked = 0
    for row in rows:
        for sub, suffix, key in [("tif", ".tif", "image_sha256"), ("csv", ".csv", "csv_sha256")]:
            if sha256(ROOT / "data/raw" / sub / (row["sample_id"]+suffix)) != row[key]:
                raise ValueError("Raw input changed: " + row["sample_id"])
            checked += 1
    old_lock = json.loads((ROOT / "outputs/next_stage_v2/00_protocol/lock.json").read_text())
    if sha256(ARCHIVE) != old_lock["archive_sha256"]:
        raise ValueError("Pooled archive changed.")
    counts = {}
    for subset, data in corpus.splits.items():
        extras = sum(len(coords)-len(np.unique(np.floor(coords).astype(int), axis=0)) for coords in data["y"])
        counts[subset] = dict(images=len(data["x"]), points=sum(map(len, data["y"])),
                              weak_truths=int(sum(map(np.count_nonzero, data["weak"]))), same_pixel_collisions=extras)
        if extras:
            raise ValueError("Dense target collision requires a protocol decision.")
    write_json(OUT / "00_protocol/data_audit.json", dict(raw_hashes_verified=checked, subsets=counts,
               test_arrays_loaded=False, labels_unchanged=True, blinded_review="pending", source_group_review="inherited; biological independence unverified"))
    # Train-only, predeclared shift diagnostic. Normalize once; wrap/crop effects
    # remain a limitation because SE has global pooling, even in the common ROI.
    data = corpus.splits["train"]
    rng = np.random.default_rng(20261026)
    selected = []
    for size in (256, 512):
        choices = np.flatnonzero(data["original_shape"][:, 0] == size)
        selected += rng.choice(choices, 5, replace=False).tolist()
    model, _ = predict._load_model(ROOT / "outputs/261024_clean220_random/seed_42/best_model.pt", torch.device(device))
    results = []
    with torch.no_grad():
        for index in selected:
            h, w = map(int, data["original_shape"][index])
            truth = data["y"][index]
            truth = truth[((truth >= 67).all(1)) & (truth[:, 0] < h-67) & (truth[:, 1] < w-67)]
            for shift in (0, 1, 2, 3):
                image = np.zeros((512, 512), dtype=np.float32)
                image[:h, :w] = np.roll(data["x"][index, :h, :w], (shift, shift), axis=(0, 1))
                tensor = torch.from_numpy(image[None, None]).to(device)
                output = model(tensor)[0].float().cpu().numpy()
                coords, scores = decode_candidates(output, (h, w))
                coords = coords[scores > 0.12678571428571428] - shift
                coords = coords[(coords >= 67).all(1) & (coords[:, 0] < h-67) & (coords[:, 1] < w-67)]
                match = match_coordinates(coords, truth, 3.)
                results.append(dict(sample_id=str(data["sample_id"][index]), size=h, diagonal_shift_px=shift,
                                    n_truth=len(truth), tp=match.tp, fp=match.fp, fn=match.fn))
    write_csv(OUT / "00_protocol/train_shift_audit.csv", results)
    write_json(OUT / "00_protocol/train_shift_audit.json", dict(n_train_images=10, shifts=[0, 1, 2, 3],
                diagonal_translation=True, normalization="once_before_translation", common_roi_margin=67,
                limitations="Wrap boundaries and global SE prevent causal attribution to grid phase; train-only descriptive diagnostic",
                threshold="frozen v2 seed42 threshold; no new tuning", test_evaluated=False))
    print("AUDIT COMPLETE", json.dumps(counts), flush=True)


def pilot(corpus, device):
    data = corpus.splits["train"]
    indices = []
    for size in (256, 512):
        indices += np.flatnonzero(data["original_shape"][:, 0] == size)[:4].tolist()
    results = []
    for key in NAMES:
        reset_rng(999)
        model = make_model(key, device).to(memory_format=torch.channels_last)
        dataset = DevelopmentDataset(data, key == "H0")
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        if device == "xpu":
            torch.xpu.reset_peak_memory_stats()
        loss = native_epoch(model, loader(dataset, 999, indices), dense_heatmap_loss if key == "H0" else combined_dice_rmse_loss, device, optimizer)
        model.eval()
        tensor = torch.from_numpy(data["x"][indices[-1]][None, None]).to(device).contiguous(memory_format=torch.channels_last)
        mask = torch.ones_like(tensor)
        durations = []
        with torch.no_grad():
            for iteration in range(25):
                if device == "xpu": torch.xpu.synchronize()
                tick = time.perf_counter()
                prediction = model(tensor, valid_mask=mask) if getattr(model, "requires_valid_mask", False) else model(tensor)
                if device == "xpu": torch.xpu.synchronize()
                if iteration >= 5: durations.append((time.perf_counter()-tick)*1000)
                if not torch.isfinite(prediction).all(): raise FloatingPointError("Pilot output non-finite")
        results.append(dict(key=key, parameters=sum(p.numel() for p in model.parameters()),
                   native_loss=loss, mixed_batch_sizes=[256]*4+[512]*4,
                   float32_forward_median_ms=float(np.median(durations)),
                   peak_allocated_mib=torch.xpu.max_memory_allocated()/2**20 if device == "xpu" else None))
        print("PILOT", json.dumps(results[-1]), flush=True)
        del model, optimizer, dataset, prediction, tensor, mask
        if device == "xpu": torch.xpu.empty_cache()
    base = results[0]
    for row in results:
        row["parameter_fraction_above_R0"] = row["parameters"]/base["parameters"]-1
        row["forward_time_ratio_to_R0"] = row["float32_forward_median_ms"]/base["float32_forward_median_ms"]
    write_json(OUT / "00_protocol/runtime_pilot.json", dict(results=results, repeats=20, warmups=5,
                no_full_training_performed=True, forward_only=True, output_size_differs_for_H0=True))


def report():
    rows = []
    for path in sorted((OUT / "01_screen").glob("*/seed_*/selection.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        config = json.loads(path.with_name("registered_config.json").read_text(encoding="utf-8"))
        rows.append(dict(key=config["key"], seed=config["seed"], **result["primary"],
                         native_secondary_f1=result["native_secondary"]["f1"],
                         selection_delta_f1=result["selection_delta_f1"],
                         tp_median_error_px=result["tp_median_error_px"],
                         fp_farther6_from_truth=result["fp_farther6_from_truth"]))
    write_csv(OUT / "validation_comparison.csv", rows)
    write_json(OUT / "validation_comparison.json", rows)
    seed42 = {r["key"]: r for r in rows if r["seed"] == 42}
    if "R0" not in seed42:
        return
    base = seed42["R0"]
    eligible = []
    decisions = []
    for key in ("R1", "R2"):
        if key not in seed42: continue
        row = seed42[key]
        gate = row["f1"]-base["f1"] >= .005 and row["weak_recall"]-base["weak_recall"] >= -.02
        if gate: eligible.append(key)
        epoch_dir = OUT / "01_screen" / key / "seed_42" / f"epoch_{row['epoch']:03d}"
        frontier = json.loads((epoch_dir / "threshold_metrics.json").read_text(encoding="utf-8"))
        matched_fp = max((r for r in frontier if r["fp"] <= base["fp"]),
                         key=lambda r: (r["tp"], r["precision"], r["threshold"]))
        weak_gain = matched_fp["weak_recall"] - base["weak_recall"]
        decisions.append(dict(key=key, f1_delta=row["f1"]-base["f1"], weak_recall_delta=row["weak_recall"]-base["weak_recall"],
                              screening_gate_passed=gate, fp_fractional_reduction=(base["fp"]-row["fp"])/max(1, base["fp"]),
                              reference_fp_budget=base["fp"], fp_budget_operating_point=matched_fp,
                              weak_recall_gain_at_reference_fp_budget=weak_gain,
                              mechanism_target_passed=(weak_gain >= .05 if key == "R1" else
                                  (base["fp"]-row["fp"])/max(1, base["fp"]) >= .10 and row["weak_recall"]-base["weak_recall"] >= -.02)))
    pilot_path = OUT / "00_protocol/runtime_pilot.json"
    timings = {r["key"]: r["float32_forward_median_ms"] for r in json.loads(pilot_path.read_text())["results"]} if pilot_path.exists() else {}
    chosen = max(eligible, key=lambda key: (seed42[key]["f1"], -timings.get(key, float("inf")))) if eligible else None
    pending_context = (all(key in seed42 for key in ("R0", "H0", "R1", "S1")) and base["fp"] > 0 and base["fn"] > 0
                       and (seed42["R1"]["fp"] > .9*base["fp"] or seed42["R1"]["f1"]-base["f1"] < .005))
    write_json(OUT / "decision.json", dict(protocol_id=PROTOCOL, trials=decisions, selected_route=chosen,
                r2_stage_enabled=pending_context, r2_rationale="R0 still has annotation-relative false positives and misses; inspect results/annotations before conditional stage",
                weak_recall_constraint=-.02, f1_gain_gate=.005, test_evaluated=False,
                core_contribution_established=False, note="Single-seed screening and simple controls do not establish novelty or independent gain"))
    lines = ["# 第二轮模型优化：开发阶段汇报", "", "协议：pooled220_methods_v3。所有分数均为固定22张验证图上的开发成绩；没有读取或评估测试数组。",
             "", "训练图176张；固定100轮、batch8、seed42筛查；每5轮保留快照，20份快照各31阈值，最多620个选模组合。验证成绩有选择乐观偏差，不能视为独立测试。",
             "", "| 模型 | seed | 选中epoch | 点F1 | Precision | Recall | 弱点Recall | FP | FN | 原生损失选模F1 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['key']} | {row['seed']} | {row['epoch']} | {row['f1']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | {row['weak_recall']:.4f} | {row['fp']} | {row['fn']} | {row['native_secondary_f1']:.4f} |")
    lines += ["", "R0为原DeepBlink的新配对参考；H0为密集heatmap/offset联合表示参考；R1/S1检验细节直达及位置选择；R2/S2检验上下文范围选择及固定融合。",
              "", "相对R0的结构筛查关口为F1至少增加0.005且弱点Recall下降不超过0.02。R1的机制目标还要求相近FP预算下弱点Recall增加至少0.05；R2要求FP减少至少10%且保持弱点约束。这些门槛不是显著性标准。",
              "", f"当前入选复核路线：{chosen or '尚无'}。详细机制工作点、简单控制及条件性R2决策见decision.json；不得仅凭单种子上涨宣称核心贡献已建立。",
              "", "新采样器、选模及阈值规则的收益单独记录；v2与v3分数不能直接归因为结构。人工盲法标注复核仍待完成，当前点表为TrackMate参考标注；来源容器关联不等于已验证的生物学独立。",
              "", "旧52次研究估算的剩余部分挂起；本轮最多6次筛查和6次种子扩展。所有原始文件、旧权重与旧结果均保留。配置、源码、样本顺序、恢复状态、20个快照、曲线及验证候选集位于01_screen各trial目录。"]
    for key in NAMES:
        values = [r["f1"] for r in rows if r["key"] == key]
        if len(values) == 3:
            lines += ["", f"{key}三种子验证F1均值±样本SD：{np.mean(values):.6f} ± {np.std(values, ddof=1):.6f}。"]
    (OUT / "阶段汇报.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["audit", "pilot", "train", "report"])
    parser.add_argument("--models", nargs="+", choices=list(NAMES), default=["R0", "H0", "R1", "S1"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--device", choices=["cpu", "xpu"], default="xpu")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    if args.action == "report":
        report(); return
    lock_protocol()
    if args.action == "train":
        if not (OUT / "00_protocol/data_audit.json").exists() or not (OUT / "00_protocol/runtime_pilot.json").exists():
            raise ValueError("Audit and runtime pilot are required before training.")
        if any(seed not in (42, 43, 44) for seed in args.seeds): raise ValueError("Unregistered seed")
        if any(key in ("R2", "S2") for key in args.models):
            decision = json.loads((OUT / "decision.json").read_text(encoding="utf-8"))
            if not decision["r2_stage_enabled"]: raise ValueError("Conditional context stage not enabled")
        if any(seed != 42 for seed in args.seeds):
            decision = json.loads((OUT / "decision.json").read_text(encoding="utf-8"))
            route = decision["selected_route"]
            control = dict(R1="S1", R2="S2").get(route)
            if not route or not set(args.models) <= {"R0", route, control}:
                raise ValueError("Only R0 and one selected route/control may get extra seeds.")
    corpus = DevelopmentCorpus()
    if args.action == "audit": audit(corpus, args.device)
    elif args.action == "pilot": pilot(corpus, args.device)
    else:
        for key in args.models:
            for seed in args.seeds:
                run_trial(key, seed, corpus, args.device, args.resume)
                report()


if __name__ == "__main__":
    from training_wake_guard import TrainingWakeGuard
    with TrainingWakeGuard():
        main()
