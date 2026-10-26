"""Descriptive analysis of locked test matches; no inference or parameter tuning."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "deepblink_pytorch"))
sys.path.insert(0, str(ROOT / "deepblink_pytorch" / "scripts"))
from error_context_diagnostics import _extract_patch_features

rows = []
for path in sorted((OUT / "seed_42/test/matches").glob("*.csv")):
    image = tifffile.imread(ROOT / "data/raw/tif" / (path.stem + ".tif"))
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for event in csv.DictReader(handle):
            status = event["match_status"].upper()
            prefix = "true" if status == "FN" else "pred"
            y, x = float(event[prefix + "_y_px"]), float(event[prefix + "_x_px"])
            rows.append(dict(sample_id=path.stem, status=status,
                             probability=float(event["pred_p"]) if event["pred_p"] else None,
                             **_extract_patch_features(image, y, x, radius=10)))
with (OUT / "error_features.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
summary = {}
for status in ("TP", "FP", "FN"):
    events = [r for r in rows if r["status"] == status]
    summary[status] = dict(n=len(events))
    for key in ("local_zscore", "peak_offset_px", "probability"):
        values = [r[key] for r in events if r[key] is not None and np.isfinite(r[key])]
        summary[status]["median_" + key] = float(np.median(values)) if values else None
summary["interpretation"] = "Posthoc descriptive analysis only; TP/FP patches centered on prediction, FN on truth; no causal or biological identity inference."
(OUT / "error_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
