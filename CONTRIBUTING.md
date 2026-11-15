# Contributing

Use an issue to describe a failure, a reproducibility gap, or a proposed change.
Include the command, full traceback, Python/PyTorch/CUDA versions and the
relevant configuration. Do not upload restricted raw data, credentials or
private review records.

Run `python -m unittest discover -s tests -v` before submitting a pull request.
Use `python scripts/validate_release.py` to check release assets and hashes.
CUDA-dependent experiments additionally require the documented GPU smoke test.

Keep core mathematics, data protocols and execution optimizations distinguishable.
For experiment changes, report the seed, split, target indices, timing boundary
and unfavorable outcomes. Do not replace historical benchmark numbers with new
results without retaining their provenance and naming the changed protocol.
