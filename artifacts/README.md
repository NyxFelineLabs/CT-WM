# Frozen research weights

Weights are attached to the GitHub release, not committed to Git history:

```bash
python scripts/fetch_weights.py
```

`configs/weights_manifest.json` pins every public asset by SHA-256 and expected
destination. `--verify-only` performs an offline integrity check.

The L=32, L=16, L=64 and no-latent world models retain their trained tensor
values. Packaged checkpoints contain only the model state dictionary, scaler
statistics and nonidentifying model metadata; absolute training-machine paths
are removed. Consequently the container file hash differs from the historical
checkpoint, while the tensor fingerprint is retained in the manifest.
SAC/PPO files are the previous frozen terminal policies (50,000 interactions),
not retrained models. Source hashes are recorded alongside release hashes.

These are experimental checkpoints for the documented dataset/task. They do not
establish independent-network generalization or online safety. Download only
the assets pinned by the manifest; SB3 policy archives use Python serialization.
