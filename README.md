# Topology-Aware Dynamics (TAD)

Implementation of the research specification *Topology-Aware Prediction of Neural
Network Training Dynamics* (`topology_aware_dynamics_spec_v2.md`).

TAD tests whether **network topology and functional state predict the rotation,
scaling, coordination, and functional evolution of future training updates beyond
what tuned momentum, online subspace tracking, and current-gradient methods
already capture** — and whether that residual predictability can improve training
after full compute amortization.

The central comparison is always against **strong history-only baselines**
(tuned EMA, VAR, DMD/Koopman, online subspace tracking, periodic SVD, and the
analytic deep-linear contractions), never mere persistence. A learned predictor
earns credit only for the residual `R_{t+h} = Y_{t+h} - Ŷ_baseline`.

## Install (uv)

```bash
uv venv
uv sync                 # installs torch, zarr, h5py, safetensors, ...
uv pip install -e .
```

All commands below can be prefixed with `uv run`.

## The E1 pipeline (spec Sections 21–23)

```bash
# 1. Generate trajectories (one per seed), with dense discovery-mode logging
tad generate-trajectories --config configs/experiments/e1.yaml

# 2. Build the supervised dynamics dataset (windows split by whole run)
tad build-dataset --config configs/experiments/e1.yaml

# 3. Train + evaluate all predictors and baselines across targets/horizons
tad train-predictor --config configs/experiments/e1.yaml
tad evaluate-predictor --config configs/experiments/e1.yaml

# 4. Per-trajectory temporal-structure analysis (Section 13)
tad analyze-temporal-structure --run-dir artifacts/experiments/e1_two_layer_stationary/trajectories/e1_two_layer_stationary__seed0

# 5. Oracle candidate-update selection + predicted-subspace optimizer (Section 14)
tad run-online-optimizer --config configs/experiments/e1.yaml

# 6. Assemble the first report (Section 22)
tad make-report --experiment-dir artifacts/experiments/e1_two_layer_stationary
```

`tad run-e1 --config <cfg>` runs the whole chain. A fast end-to-end sanity
configuration is `configs/experiments/e1_smoke.yaml`.

## What each stage produces

| Stage | Artifact |
|---|---|
| `generate-trajectories` | `trajectories/<run>/` — `tensors.zarr` (chunked tensors), `scalars.parquet`, `probes.npz`, `checkpoints/*.safetensors`, `manifest.json`, `checksums.json`, resolved config + provenance |
| `build-dataset` | `processed/<target>__H<hist>__h<horizon>.npz` + `dataset_manifest.json` (split by whole run, leakage-checked) |
| `train/evaluate-predictor` | `predictor_eval.json` — nMSE / cosine / R² / subspace overlap / energy capture + skill-vs-baseline per predictor × target × horizon |
| `analyze-temporal-structure` | `temporal_structure.json` — autocorrelation, cosine/overlap/energy-capture vs horizon, analytic-contraction check |
| `run-online-optimizer` | `oracle_results.json`, `subspace_optimizer.json` |
| `make-report` | `report/report.md` + PNGs (the 17 required items, with per-hypothesis conclusions) |

## Layout

See `topology_aware_dynamics_spec_v2.md` Section 18. Source lives in `src/tad`:
`data/` (generators), `models/` (deep linear, nonlinear MLP, residual linear,
transformer gate), `training/` (trainer, optimizers, replay, candidate updates),
`logging/` (schema, zarr writer/reader, probes, spectral), `topology/`
(products, invariants, graph, R0–R7 representations), `datasets/` (windowing),
`predictors/` (baselines + learned), `losses/`, `evaluation/` (offline, temporal,
online optimizer, reports), `utils/`.

## Status vs spec milestones

- **Milestone 1** (correct synthetic training + logging, analytic/invariant
  tests, deterministic replay): implemented and tested.
- **Milestone 2** (minimal linear crux: strong baselines, probe/topology
  representations, residualized prediction, held-out teacher split): implemented.
- **Milestone 3** (immediate nonlinear falsification): `configs/experiments/e2_nonlinear.yaml`
  with local-Jacobian / activation-gating capture.
- **Milestones 4–7** (drift/heterogeneity, depth/graph, oracle/amortized
  optimization, residual-MLP/transformer): generators, configs, and the
  transformer gate are scaffolded; the transformer training path is intentionally
  deferred per spec Section 23.

Tests: `pytest` (analytic gradients, product dynamics, replay determinism,
logging schema, subspace-metric invariance, candidate isolation, no leakage).

> The project prefers a precise negative result over a vague positive one
> (spec Section 26). Negative results are retained in reports, never tuned away.
