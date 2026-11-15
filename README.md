# CT-WM

**Calibrated & Trusted World Model for O-RAN** — an open research implementation
for KPI forecasting, uncertainty diagnosis, and short-horizon offline planning.

[中文说明](README.zh-CN.md) · [Experiment guide](docs/EXPERIMENTS.md) ·
[Recorded results](docs/RESULTS.md) ·
[Data access](data/README.md) · [Attribution](NOTICE)

CT-WM brings together probabilistic KPI forecasting, interval calibration and
compute-efficient action screening for O-RAN decision support. A multi-scale
state-space predictor supplies future KPI distributions; lightweight MPPI
compares candidate actions, and an external safety gate supports joint PRB/MCS
experiments. The Python evaluation loop includes local KPM/RC messages, OLLA
and frozen SAC/PPO policies.

### CT-WM design focus

- **Calibrated prediction:** validation-fitted interval scales, with coverage
  and width evaluated separately from action selection.
- **Budget-aware planning:** reduced-population, eight-step MPPI with frozen
  prediction weights and measured decision latency.
- **Risk evaluation:** joint PRB/MCS control, external fallback, and matched
  long-horizon, ablation and multi-SNR comparisons.

These are framework and evaluation features; the prediction backbone and base
training objective reuse prior work, credited in [Attribution](third_party/README.md).
“Trusted” names the evaluation objective, not a certification or a proven online
safety guarantee.

## What is included

- Multi-scale SSM/latent prediction backbone and unchanged training objective.
- File-isolated **7/1/2** splits, train-only scaling, L=16/32/64 and no-z runs.
- Eight-component mixture intervals, single-temperature and interval-level scaling.
- Original CEM and reduced-population MPPI for eight-step offline screening.
- Python-only KPM → xApp → RC loop; PRB/MCS controllers and 128-step extensions.
- Aggregate experiment results, provenance, tests and SHA-256-pinned release checkpoints.

Raw third-party CSVs, review history, build logs, ns-3, C++ code and RIC containers
are not bundled. The local JSON messages are **not ASN.1 E2SM/E2AP** or a live RIC.

## Quick start

Use Python 3.10+ in your existing environment. CUDA is required for the graph-based
planners; CPU suffices for interface tests and the small synthetic demo.

```bash
git clone https://github.com/NyxFelineLabs/CT-WM.git
cd CT-WM
python scripts/check_environment.py --rl
# Reuse your working PyTorch/CUDA installation; install only missing packages.
python -m pip install -e . --no-deps --no-build-isolation
python scripts/demo.py --steps 16
python -m unittest discover -s tests -v
```

The demo uses explicitly synthetic inputs and reports no benchmark claims.
`check_environment.py` only inspects packages; it never installs or changes them.
The optional RL modules require `gymnasium` and `stable-baselines3` (recorded runs
used 2.7.1). Plotting requires Matplotlib. No virtual environment is created by
any project script.

### Run with the recorded dataset and frozen weights

1. Obtain the [IEEE DataPort dataset](https://doi.org/10.21227/v8p4-v974) under its
   access terms and keep all ten CSVs separate in `data/raw/`.
2. Download the pinned weights and prepare file-isolated windows:

```bash
python scripts/fetch_weights.py
python scripts/prepare_data.py --window 32 --check-hashes
python scripts/smoke_gpu.py
python experiments/e2_experiment.py verify
```

The weights include L=16/32/64/no-z predictors and the previously trained SAC/PPO
terminal policies. This path performs **no retraining**. See the
[experiment guide](docs/EXPERIMENTS.md) for prediction, calibration, 8-step MPPI,
128-step RHC, robustness and ablation commands.

## Recorded results at a glance

These are existing measured runs, not estimates from the CPU demo. Each protocol
has its own reference, timing boundary and environment.

| Evaluation | Recorded CT-WM result | Scope |
|---|---|---|
| One-step RSRP | MAE **0.135317 dB**, RMSE **0.264605 dB** | 11,867 common held-out labels, L=32, 478,886 parameters |
| RSRP interval coverage | **51.344% / 95.601%** | Nominal 50% / 95%, mixture quantiles, validation-fitted scales |
| Eight-step offline planning | PRB saving **17.851%**, SE change **+1.577%** | 48 finite model-internal episodes; paired hold reference |
| Eight-step decision latency | Mean **12.284 ms**, P95 **14.945 ms** | Warm CPU-window-to-CPU-action measurement on RTX 3090 |
| Joint 128-step extension | SE **+0.478%**, PRB saving **−0.572%**, worst trajectory **−4.778%** | 48 paired Python-link trajectories, OLLA reference |

Prediction improves over the local persistence baseline, but short-horizon
control gains are evaluated inside the same frozen world model. They do not
establish real counterfactual network gains. The long-horizon extension does not
demonstrate simultaneous resource savings and SE improvement; matched persistence
control and multi-SNR failure cases are retained in [the results](docs/RESULTS.md).
Offline interval calibration does not transfer automatically to that different
closed-loop mapping. A 15 ms operator-boundary budget is not a preemptive hard
deadline.

## Layout

```text
src/wm_ms3m/      Attributed prediction/training core
experiments/     Forecasting, interval evaluation, MPPI, OLLA and xApp experiments
scripts/         Data preparation, weight download, demos and validation
configs/         Frozen validation-selected configurations and asset hashes
artifacts/       Model configurations; downloaded weights are ignored by Git
data/            Data access instructions and source-file fingerprints
results/         Historical aggregate results and provenance
docs/            Usage, experiment protocols and result interpretation
tests/           OLLA, message-loop, split-boundary and executor-equivalence tests
third_party/     Original MIT notice and compatibility diff
```

New outputs go to ignored `runs/`; historical aggregate tables in `results/`
remain separate. Do not mix throughput per sample with online decision latency.

## License and citation

Code is MIT-licensed; see [LICENSE](LICENSE), [NOTICE](NOTICE) and the
[third-party notices](third_party/README.md). Raw data retain their source terms.
Release weights are provided under MIT with the same attribution notices. This does not grant access rights to the raw dataset.

```bibtex
@software{nyxfelinelabs_ctwm_2026,
  author = {{NyxFelineLabs}},
  title = {{CT-WM}: World-Model Prediction and Short-Horizon Planning for O-RAN},
  year = {2026},
  version = {0.1.0},
  url = {https://github.com/NyxFelineLabs/CT-WM}
}
```

When using the prediction backbone, also cite its original publication listed
in [Attribution and references](third_party/README.md).
