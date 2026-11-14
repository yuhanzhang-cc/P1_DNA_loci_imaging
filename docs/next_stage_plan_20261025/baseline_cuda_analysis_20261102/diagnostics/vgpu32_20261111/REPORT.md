# Separate vGPU32 handoff verification — 2026-11-11

The new package is locally ready. No formal training, CUDA execution or new performance estimate was produced. The original 5090 package and the approved nine-run study remain available unchanged. Start instructions: [AUTODL_START.md](AUTODL_START.md).

| Item | Verified result |
|---|---|
| Upload ZIP | `outputs/next_stage_v4_recipe_20261101/22_vgpu32_handoff_20261111/p1_vgpu32_training_20261111.zip` |
| Bytes / members | 288,367,301 / 466 |
| SHA256 | `b4f4bb9beb8ef036354cdff7db1aa72206df345684c36b084134a5b02092d684` |
| Original 5090 ZIP SHA256, unchanged | `130375662508e59f8407e91a8b37db8257ed77e7462e200a49abb47a4e6ede8e` |
| Unit and regression checks | 130 CPU tests + 22 scaffold tests = 152 passed |
| Fresh ZIP extraction | CRC, sealed sources, child manifests and original human-file bytes passed |
| Frozen S0/P0 configurations | Byte-identical to original approved 5090 package |
| Protected historical files | 362 / 362 unchanged |
| Source files in root and mechanism child | 119 / 119 byte-identical to sealed worktree sources |
| Shell syntax | All 9 root scripts passed `bash -n` |
| Formal tasks | S0/P0 six + scratch three, all `not_started`, zero real updates |
| Packaged CPU readiness probe | One distinct synthetic step, zero real-data updates; CUDA remains unverified |
| Real vGPU preflight | Pending actual rented hardware; no CPU substitute |

Evidence: [CHECKS.json](CHECKS.json), [RELEASE.json](RELEASE.json), `evidence/unit_cpu.log`, `evidence/unit_scaffolds.log`, `evidence/sealed_verify.log`, `evidence/hardware_dryrun.log`, `evidence/status.log`, `evidence/dry_S0.log`, `evidence/dry_P0.log`, `evidence/packaged_cpu_probe.log` and `evidence/syntax_*.log`.

## Scope and interpretation

The new track has an explicit Ada/vGPU32 profile, pinned torch2.11.0+cu128 / CUDA12.8, all three historical initializer hashes, a locked resume identity, and a mandatory zero-update fixed-checkpoint transfer check. All 14 frozen references must reproduce validation TP/FP/FN and per-image prediction counts, with a maximum one-to-one coordinate difference of 0.005px. Scores are recorded; thresholds are never recalibrated in this check. The tolerance was set before any vGPU readout and must not be adjusted to admit an observed failure.

Passing this check supports using the new hardware for the planned development track; it does not prove equal training trajectories. P0 versus S0 remains a same-track contrast. S0 versus historical H0 and comparisons to historical R0/G0/Spotiflow_F carry a cross-hardware qualification in analysis tables, the manifest and English manuscript output. Full strict replication of the original hardware study should use the preserved 5090 package in a fresh directory. No extra H0 runs are authorized here.

The hardware profile and transfer evidence are included in the generated delivery archive, and hardware context propagates to the frozen evaluation package. If a lean mechanism return is analyzed locally, its copied hardware profile still preserves the cross-hardware qualification. No architecture, training recipe, frozen baseline number, human response or research decision was edited.

## Reproduction

From a checkout with the private development inputs and frozen references restored, build into a new, nonexistent output path:

```powershell
python deepblink_pytorch/scripts/diagnostics/completion_v1/cli.py build --root . --review-dir outputs/next_stage_v4_recipe_20261101/21_approved_handoff_20261111/received --hardware-profile docs/next_stage_plan_20261025/baseline_cuda_analysis_20261102/diagnostics/vgpu32_20261111/hardware_profile.json --output "<NEW_KIT>"
```

The generated directory can be verified and planned without CUDA:

```bash
python3 deepblink_pytorch/scripts/diagnostics/completion_v1/cli.py cloud-verify --root .
python3 deepblink_pytorch/scripts/diagnostics/completion_v1/cli.py hardware-preflight --root .
python3 deepblink_pytorch/scripts/diagnostics/completion_v1/cli.py cloud-plan --root .
```

Focused synthetic tests in a development environment with pytest, torch and the tracked dependencies:

```bash
python -m pytest -q deepblink_pytorch/scripts/diagnostics/completion_v1/test_hardware_policy.py
```

Exact local commands and successful terminal outputs are retained in `evidence/*.log`. The local private builder/verification orchestrator is `outputs/next_stage_v4_recipe_20261101/22_vgpu32_handoff_20261111/prepare_verify.py`; it deliberately refuses to overwrite its output/evidence directories. Do not repeat it against the sealed release path.

Review the diff for explicit hardware admission, unchanged historical initialization enforcement, comparison labels, and the new runbook. Dependencies, `round2_protocol.py`, network modules and old result artifacts are unchanged. Private ZIPs, images, weights and per-case review files stay outside Git.
