"""First-report generation (Sections 22, 17).

Assembles training curves, temporal-structure plots, prediction-metric tables,
residual/topology comparisons, oracle results, compute accounting, and an
auto-derived per-hypothesis conclusion section into a Markdown report with PNGs.
Everything is defensive: whatever artifacts exist get included.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..logging.reader import TrajectoryReader
from . import temporal_structure as ts


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _training_curves(reader, outdir) -> str:
    s = reader.scalars()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(s["step"], s["train_loss_before"], label="train loss", lw=1)
    val = s.dropna(subset=["validation_loss"])
    if len(val):
        ax.plot(val["step"], val["validation_loss"], "o-", label="val loss", ms=3)
    ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_ylabel("MSE"); ax.legend()
    ax.set_title("Training curves")
    p = outdir / "01_training_curves.png"; _save(fig, p); return p.name


def _norm_trajectories(reader, outdir) -> str:
    s = reader.scalars()
    fig, ax = plt.subplots(figsize=(6, 4))
    for col, lab in [("gradient_global_norm", "grad"), ("update_global_norm", "update"),
                     ("parameter_global_norm", "param")]:
        if col in s.columns:
            ax.plot(s["step"], s[col], label=lab, lw=1)
    ax.set_yscale("log"); ax.set_xlabel("step"); ax.legend(); ax.set_title("Global norms")
    p = outdir / "02_norm_trajectories.png"; _save(fig, p); return p.name


def _singular_trajectories(reader, outdir) -> str:
    fig, axes = plt.subplots(1, len(reader.layer_names), figsize=(5 * len(reader.layer_names), 4), squeeze=False)
    for j, layer in enumerate(reader.layer_names):
        sv = ts.singular_value_trajectories(reader, layer)
        steps = reader.steps_for(f"layer__{layer}__top_singular_values_W")
        for k in range(sv.shape[1]):
            axes[0, j].plot(steps, sv[:, k], lw=0.8)
        axes[0, j].set_title(f"{layer} singular values"); axes[0, j].set_xlabel("step")
    p = outdir / "03_singular_trajectories.png"; _save(fig, p); return p.name


def _effective_rank(reader, outdir) -> str:
    fig, ax = plt.subplots(figsize=(6, 4))
    for layer in reader.layer_names:
        er = ts.effective_rank_trajectory(reader, layer)
        ax.plot(reader.steps_for(f"layer__{layer}__W"), er, label=layer)
    ax.set_xlabel("step"); ax.set_ylabel("effective rank"); ax.legend()
    ax.set_title("Effective rank (energy convention)")
    p = outdir / "04_effective_rank.png"; _save(fig, p); return p.name


def _cosine_vs_horizon(reader, horizons, outdir) -> str:
    fig, ax = plt.subplots(figsize=(6, 4))
    for layer in reader.layer_names:
        gc = ts.gradient_cosine_vs_horizon(reader, layer, horizons)
        uc = ts.update_cosine_vs_horizon(reader, layer, horizons)
        ax.plot(list(gc), list(gc.values()), "o-", label=f"{layer} grad")
        ax.plot(list(uc), list(uc.values()), "s--", label=f"{layer} update")
    ax.set_xlabel("horizon"); ax.set_ylabel("cosine"); ax.set_xscale("log"); ax.legend(fontsize=7)
    ax.set_title("Gradient/update cosine vs horizon")
    p = outdir / "05_cosine_vs_horizon.png"; _save(fig, p); return p.name


def _subspace_overlap(reader, horizons, outdir) -> str:
    fig, ax = plt.subplots(figsize=(6, 4))
    for layer in reader.layer_names:
        ov = ts.subspace_overlap_vs_horizon(reader, layer, horizons)
        ax.plot(list(ov), list(ov.values()), "o-", label=layer)
    ax.set_xlabel("horizon"); ax.set_ylabel("overlap"); ax.set_xscale("log"); ax.legend()
    ax.set_title("Gradient subspace overlap vs horizon")
    p = outdir / "06_subspace_overlap.png"; _save(fig, p); return p.name


def _energy_capture(reader, horizons, outdir) -> str:
    layer = reader.layer_names[0]
    ec = ts.energy_capture_vs_horizon(reader, layer, horizons)
    fig, ax = plt.subplots(figsize=(6, 4))
    for r, d in ec.items():
        ax.plot(list(d), list(d.values()), "o-", label=f"rank {r}")
    ax.set_xlabel("horizon"); ax.set_ylabel("energy captured"); ax.set_xscale("log"); ax.legend()
    ax.set_title(f"Future-gradient energy captured by current subspace ({layer})")
    p = outdir / "07_energy_capture.png"; _save(fig, p); return p.name


def _probe_autocorr(reader, horizons, outdir) -> str:
    layer = reader.layer_names[0]
    wp = reader.layer_field(layer, "probe_WP")
    ac = ts.autocorrelation(wp, [h for h in horizons if h < len(wp)])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(list(ac), list(ac.values()), "o-")
    ax.set_xlabel("horizon"); ax.set_ylabel("autocorrelation"); ax.set_xscale("log")
    ax.set_title(f"Probe-action autocorrelation ({layer})")
    p = outdir / "08_probe_autocorr.png"; _save(fig, p); return p.name


def _predictor_bars(eval_results, outdir) -> Optional[str]:
    """Bar chart of skill vs strongest baseline at h=1 (residual predictability)."""
    if not eval_results:
        return None
    # eval_results: {predictor: {target: {horizon: metrics}}}
    h_key = None
    fig, ax = plt.subplots(figsize=(7, 4))
    names, skills = [], []
    for pred, by_target in eval_results.items():
        # pick first target, smallest horizon
        for tgt, by_h in by_target.items():
            hs = sorted(int(k) for k in by_h)
            m = by_h[str(hs[0])]
            skill = m.get("skill_vs_tuned_ema", m.get("r2"))
            names.append(f"{pred}\n{tgt} h{hs[0]}")
            skills.append(skill)
            break
    ax.bar(range(len(names)), skills)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=6)
    ax.axhline(0, color="k", lw=0.8); ax.set_ylabel("skill vs tuned EMA")
    ax.set_title("Residual predictability beyond strongest baseline")
    p = outdir / "09_predictor_skill.png"; _save(fig, p); return p.name


def _hypothesis_conclusions(reader, eval_results, oracle) -> Dict[str, str]:
    """Heuristic, auto-derived per-hypothesis read (Section 17 / 22.17)."""
    out = {}
    layer = reader.layer_names[0]
    # H3: subspace persists -> high energy capture at h=1
    ec = ts.energy_capture_vs_horizon(reader, layer, [1], ranks=(8,))[8][1]
    out["H3 (slow subspace)"] = (
        f"rank-8 energy capture at h=1 = {ec:.3f}; "
        + ("subspace captures substantial future-gradient energy." if ec > 0.5
           else "weak subspace persistence.")
    )
    # H4: analytic contraction recovered
    ac = ts.analytic_contraction_check(reader)
    if ac:
        out["H4 (topology/analytic)"] = (
            f"layer-gradient contractions recovered to nMSE "
            f"G1={ac['G1_contraction_nmse']:.2e}, G2={ac['G2_contraction_nmse']:.2e} "
            "(analytic baseline validated; learned topology must beat this)."
        )
    if eval_results:
        # H1/H7: any learned model beats tuned EMA?
        beats = []
        for pred, by_target in eval_results.items():
            for tgt, by_h in by_target.items():
                for h, m in by_h.items():
                    sk = m.get("skill_vs_tuned_ema")
                    if sk is not None and sk > 0:
                        beats.append((pred, tgt, h, sk))
        out["H1/H7 (residual predictability)"] = (
            f"{len(beats)} (predictor,target,horizon) cells beat tuned EMA."
            if beats else "no learned model beat tuned EMA (negative result retained)."
        )
    if oracle:
        out["H8 (optimization value)"] = (
            f"predicted-proposal win rate {oracle.get('predicted_win_rate', 0):.2f}, "
            f"mean held-out improvement {oracle.get('mean_improvement_over_baseline', 0):.2e}."
        )
    return out


def make_report(experiment_dir, horizons=(1, 2, 4, 8, 16, 32, 64, 100)) -> Path:
    experiment_dir = Path(experiment_dir)
    outdir = experiment_dir / "report"
    outdir.mkdir(parents=True, exist_ok=True)
    horizons = list(horizons)

    # locate a trajectory
    traj_root = experiment_dir / "trajectories"
    run_dirs = [p for p in sorted(traj_root.iterdir()) if (p / "tensors.zarr").exists()] \
        if traj_root.exists() else []
    if not run_dirs:
        raise FileNotFoundError(f"no trajectories under {traj_root}")
    reader = TrajectoryReader(run_dirs[0])

    eval_results = _maybe_json(experiment_dir / "predictor_eval.json")
    oracle = _maybe_json(experiment_dir / "oracle_results.json")
    compute = _maybe_json(experiment_dir / "compute_accounting.json")

    imgs = [
        _training_curves(reader, outdir),
        _norm_trajectories(reader, outdir),
        _singular_trajectories(reader, outdir),
        _effective_rank(reader, outdir),
        _cosine_vs_horizon(reader, horizons, outdir),
        _subspace_overlap(reader, horizons, outdir),
        _energy_capture(reader, horizons, outdir),
        _probe_autocorr(reader, horizons, outdir),
    ]
    skill_img = _predictor_bars(eval_results, outdir)
    conclusions = _hypothesis_conclusions(reader, eval_results, oracle)

    md = _assemble_markdown(reader, imgs, skill_img, eval_results, oracle, compute, conclusions)
    report_path = outdir / "report.md"
    report_path.write_text(md, encoding="utf-8")
    return report_path


def _maybe_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else None


def _assemble_markdown(reader, imgs, skill_img, eval_results, oracle, compute, conclusions) -> str:
    L = []
    L.append("# E1 First Report — Topology-Aware Dynamics\n")
    L.append(f"- run_id: `{reader.run_id}`  seed: {reader.seed}  total_steps: {reader.total_steps}")
    L.append(f"- layers: {reader.layer_names}\n")
    titles = ["Training curves", "Gradient/update/param norms", "Singular-value trajectories",
              "Effective rank", "Gradient/update cosine vs horizon", "Subspace overlap vs horizon",
              "Future energy capture", "Probe-action autocorrelation"]
    for t, img in zip(titles, imgs):
        L.append(f"## {t}\n\n![{t}]({img})\n")
    if skill_img:
        L.append(f"## Residual predictability beyond strongest baseline\n\n![skill]({skill_img})\n")
    if eval_results:
        L.append("## Prediction metrics (all predictors / baselines)\n")
        L.append(_metrics_table(eval_results))
    if oracle:
        L.append("## One-step candidate-update oracle (held-out microbatch)\n")
        L.append(f"- evals: {oracle.get('n_evals')}, predicted win rate: {oracle.get('predicted_win_rate'):.3f}, "
                 f"baseline win rate: {oracle.get('baseline_win_rate'):.3f}")
        L.append(f"- mean held-out improvement over baseline: {oracle.get('mean_improvement_over_baseline'):.3e}")
        L.append(f"- mean same-batch vs held-out selection gap: {oracle.get('mean_same_vs_heldout_gap'):.3e}\n")
    if compute:
        L.append("## Compute & storage accounting\n")
        L.append("```json")
        L.append(json.dumps(compute, indent=2))
        L.append("```\n")
    L.append("## Hypothesis conclusions\n")
    for k, v in conclusions.items():
        L.append(f"- **{k}**: {v}")
    L.append("\n> Negative results are retained verbatim (interpretation rule 16).")
    return "\n".join(L)


def _metrics_table(eval_results) -> str:
    rows = ["| predictor | target | horizon | nMSE | cosine | R2 | skill vs EMA | subspace overlap |",
            "|---|---|---|---|---|---|---|---|"]
    for pred, by_target in eval_results.items():
        for tgt, by_h in by_target.items():
            for h in sorted(by_h, key=lambda x: int(x)):
                m = by_h[h]
                rows.append("| {} | {} | {} | {:.3e} | {:.3f} | {:.3f} | {} | {} |".format(
                    pred, tgt, h, m.get("nmse", float("nan")), m.get("cosine", float("nan")),
                    m.get("r2", float("nan")),
                    f"{m.get('skill_vs_tuned_ema', float('nan')):.3f}",
                    f"{m.get('subspace_overlap', float('nan')):.3f}"))
    return "\n".join(rows) + "\n"
