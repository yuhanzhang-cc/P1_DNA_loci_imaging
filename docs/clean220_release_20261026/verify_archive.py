"""Verify published clean220 artifacts without loading models or image arrays."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    root = args.root.resolve()
    manifest_path = root / "docs/clean220_release_20261026/artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = []
    for item in manifest["artifacts"]:
        path = (root / item["path"]).resolve()
        if not path.is_relative_to(root):
            failures.append({"path": item["path"], "reason": "outside repository"})
        elif not path.is_file():
            failures.append({"path": item["path"], "reason": "missing"})
        elif path.stat().st_size != item["bytes"]:
            failures.append({"path": item["path"], "reason": "size mismatch"})
        elif sha256(path) != item["sha256"]:
            failures.append({"path": item["path"], "reason": "sha256 mismatch"})
    print(json.dumps({"checked": len(manifest["artifacts"]), "failures": failures}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
