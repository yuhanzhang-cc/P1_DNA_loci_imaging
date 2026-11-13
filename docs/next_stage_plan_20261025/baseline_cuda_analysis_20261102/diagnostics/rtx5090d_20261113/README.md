# RTX5090D independent handoff — 2026-11-13

Use [AUTODL_START.md](AUTODL_START.md). This is a new package for original RTX5090D 32GB / compute capability12.0, not D v2 24GB. Both old 5090 and vGPU32 ZIPs remain unchanged. The approved nine runs, human inputs, configs and historical initialization hashes remain unchanged. No formal training or CUDA validation has run locally.

Hardware-profile admission, frozen-checkpoint parity and same-device resume locks are mandatory. P0-S0 is within-track; historical comparisons remain cross-hardware development references. Passing inference parity does not guarantee identical training or better F1. See CHECKS.json, RELEASE.json and evidence/ for actual local checks.
