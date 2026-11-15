# Release validation — v0.1.0

Validated on 2026-11-14 using the existing Ubuntu 22.04 WSL environment,
Python 3.10.12, PyTorch 2.1.2+cu121 and an NVIDIA RTX 3090.

- Six CPU unit tests passed, covering OLLA negative feedback, JSON KPM/RC
  exchange, action bounds, file-isolated splits and cached-executor equivalence.
- The synthetic CPU demonstration and Python compilation passed.
- Frozen L=32 prediction, joint MPPI and short-horizon graph MPPI loaded and
  executed on a held-out initial window without retraining.
- Both released SAC and PPO policies loaded and executed three steps each.
- The 128-step OLLA reference verification passed.
- Backbone and recorded aggregate-result SHA-256 checks passed. The staged
  Git blobs were checked as well, preserving hashes across clean checkouts.
- Release checkpoint tensors and scaling metadata match the original trained
  checkpoints; only machine-specific checkpoint paths were removed.

These checks validate packaging and execution. They do not rerun the full
historical benchmark or constitute a fresh latency measurement. Recorded
results and their evaluation boundaries are documented in [RESULTS.md](RESULTS.md).
