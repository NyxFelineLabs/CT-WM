# Experiment guide

Run commands from the repository root in the same existing interpreter used for
PyTorch. Linux/WSL with CUDA is the validated experiment environment. CUDA Graph
planners require a supported CUDA GPU; the CPU tests do not validate GPU timing.

## 1. Prepare

```bash
python scripts/check_environment.py --rl
python -m pip install -e . --no-deps --no-build-isolation
python scripts/fetch_weights.py
python scripts/prepare_data.py --window 32 --check-hashes
python scripts/smoke_gpu.py
```

See [data access](../data/README.md). No project command automatically installs
dependencies, creates a virtual environment, downloads restricted data or trains
RL policies. Required packages are declared in `pyproject.toml`. Install missing
packages explicitly while preserving an already-working PyTorch/CUDA build.

## 2. Prediction and offline interval diagnosis

```bash
python experiments/system_prediction.py
python experiments/planner_calibration.py
```

The first command loads L=32, L=16, L=64 and no-z checkpoints. It uses the same
11,867 labels at raw row indices >=64 in the last two files. Eight full prior
forwards provide target-head means, mixture variances and full-decoder KPI means.
The second fits temperatures using validation file #8 only, then scores the
saved test predictions. It never fits temperatures on test labels.

Outputs: `runs/system_validation/` and `runs/planner_optimization/`. Historical
table records remain in `results/`. CUDA-synchronized throughput uses batch 256,
eight prior samples, 10 warmups and 100 repeats. Do not call that batch-1 latency.

## 3. Eight-step finite planning tasks

```bash
# Historical baseline traces, including the original CEM; potentially slow:
python experiments/system_planning.py --name L32
# Reduced-population MPPI, paired baseline and implementation timing:
python experiments/planner_evaluate.py
```

`planner_evaluate.py` requires the first run's archived L=32 traces. Each method
has 16 starts × 3 planner seeds × 8 decisions. The initial horizon is 8 and shrinks
as the finite episode ends. Only the first action is executed before replanning.
This is not a fixed-H=8 sustained 128-step run. MPPI uses the checked-in
validation-selected configuration (64 candidates, sigma 5, temperature 0.5).

Both planning and rollout evaluation use the frozen L=32 model. Actions are
continuous PRB in [13,25]. They enter the next history after prediction, retaining
the one-frame public-model delay. MPPI also corrects physical feedback scaling
and uses cached kernels/CUDA Graphs; CEM-to-MPPI differences are not an isolated
algorithm-only ablation.

## 4. Python link and joint PRB/MCS control

```bash
python experiments/e2_experiment.py verify
python experiments/e2_experiment.py evaluate
```

Loads frozen SAC/PPO policies, never retrains. Controls are
`[PRB_adjustment, absolute_MCS_index]`. KPM/RC objects are locally serialized JSON;
no E2AP ASN.1, network transport or live RIC is involved. The PHY is an assumed
Shannon-gap/logistic BLER model with 20 ACK/NACK trials per control cycle. OLLA
uses SINR minus offset, ACK −0.1 dB and NACK +0.9 dB. MCS efficiencies and thresholds
are the documented model assumptions, not a 3GPP-standard SINR lookup table.

This command reproduces the earlier five-method run, which has a different
recorded timing/result batch from the later eight-method supplement.

## 5. Main supplement, multi-SNR and control dimensions

```bash
python experiments/e2_supplement.py freeze
python experiments/e2_supplement.py main
python experiments/e2_supplement.py snr
python experiments/e2_supplement.py temperature
python experiments/e2_control_dimensions.py
python experiments/e2_predictor_comparison.py
```

The supplement runs the eight methods without training and keeps all 48 ×128
steps per method. SNR conditions are 0,5,10,15,20 dB shifts of the exogenous SINR
proxy, with original weights and thresholds. `temperature` needs `main_steps.csv`
from the main supplement. It holds states/RNG fixed, disables the timing budget
and checks whether interval-only scaling changes actions. `e2_control_dimensions`
needs the supplement main/protocol records. The persistence comparison uses
identical MPPI/gate/reward and its own validation-only residual scale.

Only-PRB keeps MCS=14 in search and fallback. Only-MCS projects PRB increments to
zero before scoring and holds initial PRB in execution. The variant named
`no_calib` is a **compound** mean-only stress-path ablation, not temperature-only
causal evidence. Exact commands and protocol fields are retained in source.

## 6. Optional training, explicitly requested only

```bash
python experiments/run_file_split.py --width 93
python experiments/system_train.py --name no_z
python experiments/system_train.py --name L16
python experiments/system_train.py --name L64
# Optional and separate from evaluation:
python experiments/e2_experiment.py sac
python experiments/e2_experiment.py ppo
```

Training overwrites the respective local ignored checkpoints; keep release
weights elsewhere if needed. No evaluation command retrains the world model or
SAC/PPO. The model uses the existing posterior-validation checkpoint selection
and prior-only final test. L=32 is a local assumption, not a confirmed original
paper hyperparameter. Single training seeds do not establish training variance.

## Reproduction expectations

To redraw the public aggregate robustness and benefit/risk figures without raw
data or model weights, install Matplotlib if missing and run:

```bash
python scripts/render_results.py
```

This writes PNG/PDF/SVG to ignored `runs/public_figures/` using the aggregate experiment tables.

Recorded environment: WSL Ubuntu 22.04, Python 3.10.12, PyTorch 2.1.2+cu121,
RTX 3090; the RL policies used Stable-Baselines3 2.7.1. Hashes pin inputs and
weights. GPU kernels, wall-clock timeout handling and library versions can change
timings and occasional accepted actions. Report new runs as new records.

15 ms is a check at operator boundaries; submitted CUDA work is not preempted.
Mean, P99, observed maximum and full-cycle costs must stay distinct. This project
contains no independent real-radio counterfactual labels or fresh blind holdout.
