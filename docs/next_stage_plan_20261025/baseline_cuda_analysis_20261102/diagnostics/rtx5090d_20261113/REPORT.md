# RTX5090D handoff verification — 2026-11-13

Separate 32GB RTX5090D package locally ready. No actual CUDA validation, formal training or new model performance is claimed. Original 5090 and vGPU archives are preserved.

- Upload: `p1_5090d_training_20261113.zip` and its `.sha256`.
- SHA256: `9106b1cd90c803307285eaf95bcf37ac4882745dc41f1577a36cd9c988286154`.
- Bytes: 288366926; archive members: 466.
- Tests: 137 CPU + 22 scaffold = 159 passed.
- Fresh extraction/CRC/sealed root and child source hashes passed; 119 source files match each runtime copy.
- Original human inputs and frozen S0/P0 configs byte-identical; 362 protected historical files unchanged.
- All nine approved tasks remain not_started; zero real-data optimizer updates. Packaged CPU probe uses synthetic data only.
- Nine root shell scripts pass bash syntax; S0/P0 configs, hardware dry-run, composite plan and status pass.

Run [AUTODL_START.md](AUTODL_START.md). Hardware preflight on the rented card remains mandatory: original RTX5090D 32GB / CC12.0, pinned torch2.11.0+cu128, all three historic initializer hashes, and fixed-checkpoint validation parity. No threshold recalibration, additional H0 training or architecture/recipe change is authorized. D v2 24GB, regular5090,4090 and vGPU cannot silently enter this track. Same-device recovery identity remains locked. Inference parity does not prove training equivalence or improve F1 by itself.

P0-S0 is a within-track contrast. Comparisons to old H0/R0/G0/Spotiflow remain cross-hardware development references; output tables and manuscript retain this qualification. No independent test data is included.

Evidence is in CHECKS.json, RELEASE.json and evidence/*.log; source/index correspondence is in SOURCE_CORRESPONDENCE.json. Review the focused hardware profile/test changes and startup instructions in VS Code. Historical scientific conclusions and dependencies are unchanged. Private data/weights/ZIPs stay outside Git.

Rebuild to a fresh, nonexistent output directory with private inputs restored:

```powershell
python deepblink_pytorch/scripts/diagnostics/completion_v1/cli.py build --root . --review-dir outputs/next_stage_v4_recipe_20261101/21_approved_handoff_20261111/received --hardware-profile docs/next_stage_plan_20261025/baseline_cuda_analysis_20261102/diagnostics/rtx5090d_20261113/hardware_profile.json --output "<NEW_KIT>"
```

Dry-run the packaged CLI: `python3 deepblink_pytorch/scripts/diagnostics/completion_v1/cli.py hardware-preflight --root .`.
Focused tests: `python -m pytest -q deepblink_pytorch/scripts/diagnostics/completion_v1/test_hardware_policy.py`.
Exact successful local commands and outputs are in evidence/*.log. Public log line endings alone are normalized to LF; output contents are preserved.
