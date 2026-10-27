"""Registered v3 targets, candidate decoding and exact threshold accounting."""
from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment
from scipy.special import expit
from .losses import dice_score


THRESHOLDS_V3 = np.r_[0., expit(np.linspace(-9., 9., 29)), 1.].tolist()


def training_orders(n_images: int, epochs: int, seed: int) -> list[list[int]]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return [torch.randperm(n_images, generator=generator).tolist() for _ in range(epochs)]


def encode_dense_targets(coords, shape, canvas_size: int = 512) -> np.ndarray:
    """Gaussian at floor anchors; channels are heatmap, dy, dx, anchor flag."""
    height, width = map(int, shape)
    coords = np.asarray(coords, dtype=np.float32).reshape(-1, 2)
    if not np.isfinite(coords).all() or np.any(coords < 0) or np.any(coords >= [height, width]):
        raise ValueError("Dense target coordinates are outside the original image.")
    anchors = np.floor(coords).astype(int)
    if len(np.unique(anchors, axis=0)) != len(anchors):
        raise ValueError("Same-pixel collision must not be silently overwritten.")
    target = np.zeros((canvas_size, canvas_size, 4), dtype=np.float32)
    target[height:, :, 0] = -1
    target[:, width:, 0] = -1
    for coord, (y, x) in zip(coords, anchors):
        y0, y1 = max(0, y-3), min(height, y+4)
        x0, x1 = max(0, x-3), min(width, x+4)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        radius2 = (yy-y)**2 + (xx-x)**2
        gaussian = np.where(radius2 <= 9, np.exp(-radius2/2.), 0.)
        target[y0:y1, x0:x1, 0] = np.maximum(target[y0:y1, x0:x1, 0], gaussian)
        target[y, x, 1:3] = coord - [y, x]
        target[y, x, 3] = 1
    return target


def dense_heatmap_loss(target, prediction):
    valid = target[..., 0] >= 0
    if not bool(valid.any()):
        raise ValueError("Dense batch has no valid image pixels.")
    labels, logits = target[..., 0][valid], prediction[..., 0][valid]
    dice = 1 - dice_score(labels, logits.sigmoid())
    bce = F.binary_cross_entropy_with_logits(logits, labels)
    anchors = (target[..., 3] > 0) & valid
    if bool(anchors.any()):
        error2 = (prediction[..., 1:3][anchors] - target[..., 1:3][anchors]).square().sum(-1)
        rmse = torch.sqrt(error2.mean() + torch.finfo(prediction.dtype).eps)
    else:
        rmse = prediction[..., 1:3].sum() * 0
    return dice + bce + 2 * rmse


def local_peak_mask(scores: np.ndarray) -> np.ndarray:
    """3x3 maxima; equal neighbours prefer the lexicographically earlier pixel."""
    height, width = scores.shape
    padded = np.pad(scores, 1, constant_values=-np.inf)
    keep = np.isfinite(scores)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == dx == 0:
                continue
            neighbour = padded[1+dy:1+dy+height, 1+dx:1+dx+width]
            earlier = dy < 0 or (dy == 0 and dx < 0)
            keep &= (scores > neighbour) if earlier else (scores >= neighbour)
    return keep


def decode_candidates(prediction, shape, *, dense: bool = False, cell_size: int = 4):
    """Decode without a score floor and retain the originating cell's score."""
    height, width = map(int, shape)
    if not np.isfinite(prediction).all():
        raise FloatingPointError("Non-finite model prediction.")
    if dense:
        # Peak extraction never sees the artificial padded region.
        score_map = expit(prediction[:height, :width, 0])
        rows, cols = np.nonzero(local_peak_mask(score_map))
        coords = np.column_stack([rows, cols]) + prediction[rows, cols, 1:3]
        scores = score_map[rows, cols]
    else:
        rows, cols = np.indices(prediction.shape[:2])
        rows, cols = rows.ravel(), cols.ravel()
        coords = (np.column_stack([rows, cols]) + prediction[rows, cols, 1:3]) * cell_size
        scores = prediction[rows, cols, 0]
    inside = (coords >= 0).all(1) & (coords[:, 0] < height) & (coords[:, 1] < width)
    return coords[inside].astype(np.float32), scores[inside].astype(np.float32)


def threshold_mask(scores, threshold):
    return np.ones(len(scores), dtype=bool) if threshold == 0 else scores > threshold


def sweep_candidate_counts(coords, scores, truth, thresholds=THRESHOLDS_V3, radius=3.):
    """Exact max-cardinality counts; omit only columns that can never match.

    Far candidates still count toward FP. Precompute distances once per image;
    removing all-invalid columns cannot change an optimal match against cheaper
    dummy columns. The dummy penalty remains that of the full candidate set.
    """
    coords = np.asarray(coords, dtype=np.float32).reshape(-1, 2)
    truth = np.asarray(truth, dtype=np.float32).reshape(-1, 2)
    distances = np.linalg.norm(truth[:, None] - coords[None], axis=2)
    near = np.flatnonzero(np.any(distances <= radius, axis=0))
    result = []
    for threshold in thresholds:
        keep = threshold_mask(scores, threshold)
        count = int(keep.sum())
        selected = near[keep[near]]
        nt = len(truth)
        if nt and len(selected):
            penalty = (min(count, nt) + 1) * (radius + 1)
            cost = np.full((nt, len(selected)+nt), penalty, dtype=np.float64)
            d = distances[:, selected]
            cost[:, :len(selected)] = np.where(d <= radius, d, 2*penalty)
            ti, pi = linear_sum_assignment(cost)
            valid = pi < len(selected)
            ti, pi = ti[valid], pi[valid]
            valid = d[ti, pi] <= radius
            matched_truth = ti[valid]
        else:
            matched_truth = np.empty(0, dtype=int)
        tp = len(matched_truth)
        result.append((tp, count-tp, nt-tp, matched_truth))
    return result
