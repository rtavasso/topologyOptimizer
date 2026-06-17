"""TAD command-line interface (Section 19).

Commands:
  tad generate-trajectories --config ...
  tad build-dataset         --config ...
  tad train-predictor       --config ...
  tad evaluate-predictor    --config ... [--checkpoint ...]
  tad analyze-temporal-structure --run-dir ...
  tad run-online-optimizer  --config ...
  tad make-report           --experiment-dir ...
  tad run-e1                --config ...   (end-to-end convenience)

Every command persists resolved config, provenance, seeds, metrics, and
artifacts under the experiment directory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from .config import load_config, save_run_metadata, Config
from .training.trainer import train_run


def _exp_dir(cfg: Config, root: str = "artifacts/experiments") -> Path:
    d = Path(root) / cfg.experiment["name"]
    d.mkdir(parents=True, exist_ok=True)
    return d


# -- generate-trajectories ----------------------------------------------------
def cmd_generate(args):
    cfg = load_config(args.config)
    exp = _exp_dir(cfg)
    traj_dir = exp / "trajectories"
    seeds = list(cfg.experiment.get("seeds", [0]))
    out = []
    for seed in seeds:
        print(f"[generate] seed {seed} ...")
        rd = train_run(cfg, seed, traj_dir, device=args.device)
        out.append(str(rd))
    save_run_metadata(exp, cfg, extra={"command": "generate-trajectories", "runs": out})
    print(f"[generate] wrote {len(out)} trajectories to {traj_dir}")


# -- build-dataset ------------------------------------------------------------
def cmd_build_dataset(args):
    from .datasets import build_dynamics_dataset
    cfg = load_config(args.config)
    exp = _exp_dir(cfg)
    manifest = build_dynamics_dataset(cfg, exp / "trajectories", exp / "processed")
    save_run_metadata(exp, cfg, extra={"command": "build-dataset", "manifest": manifest})
    print(f"[build-dataset] {manifest['splits']}")


# -- train + evaluate predictors ---------------------------------------------
def _evaluate_all(cfg: Config, exp: Path, device: str) -> dict:
    from .datasets import load_dynamics_dataset
    from .predictors import build_predictor
    from .predictors.baselines import Persistence, TunedEMA
    from .evaluation.offline import evaluate_prediction

    processed = exp / "processed"
    ev = cfg.get("eval", {}) if hasattr(cfg, "get") else {}
    history_length = int(ev.get("history_length", 16))
    dd = cfg.dynamics_dataset
    targets = list(ev.get("targets", dd.get("targets", ["future_gradient"])))
    horizons = list(ev.get("horizons", dd.get("horizons", [1, 8, 32])))
    predictor_names = list(cfg.predictors)
    pcfg = dict(cfg.get("predictor_config", {})) if hasattr(cfg, "get") else {}
    pcfg.setdefault("device", device)

    results: Dict[str, dict] = {pn: {} for pn in predictor_names}
    for target in targets:
        for h in horizons:
            npz = processed / f"{target}__H{history_length}__h{h}.npz"
            if not npz.exists():
                continue
            bundle = load_dynamics_dataset(npz)
            mask = bundle["mask"]
            shapes = bundle["node_shapes"]
            test_Y = bundle.get("test_Y")
            if test_Y is None:
                continue
            # baseline predictions for skill scores
            base_preds = {}
            for bname, bcls in [("persistence", Persistence), ("tuned_ema", TunedEMA)]:
                b = bcls(pcfg)
                b.fit(bundle)
                base_preds[bname] = b.predict(bundle, "test")
            for pn in predictor_names:
                pred = build_predictor(pn, pcfg)
                pred.fit(bundle)
                Yhat = pred.predict(bundle, "test")
                metrics = evaluate_prediction(Yhat, test_Y, mask, shapes, target, base_preds)
                results[pn].setdefault(target, {})[str(h)] = metrics
                print(f"[eval] {pn:32s} {target:16s} h{h:<3d} "
                      f"nMSE={metrics['nmse']:.3e} skill_vs_ema={metrics.get('skill_vs_tuned_ema', float('nan')):.3f}")
    return results


def cmd_train_predictor(args):
    cfg = load_config(args.config)
    exp = _exp_dir(cfg)
    results = _evaluate_all(cfg, exp, args.device)
    (exp / "predictor_eval.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    save_run_metadata(exp, cfg, extra={"command": "train-predictor"})
    print(f"[train-predictor] wrote {exp/'predictor_eval.json'}")


def cmd_evaluate_predictor(args):
    # evaluation is fused with training of the (cheap) predictors here
    cmd_train_predictor(args)


# -- analyze temporal structure ----------------------------------------------
def cmd_analyze(args):
    from .logging.reader import TrajectoryReader
    from .evaluation.temporal_structure import full_temporal_report
    reader = TrajectoryReader(args.run_dir)
    rep = full_temporal_report(reader)
    out = Path(args.run_dir) / "temporal_structure.json"
    out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"[analyze] wrote {out}")
    if rep.get("analytic_contraction"):
        print("[analyze] analytic contraction:", rep["analytic_contraction"])


# -- online optimizer ---------------------------------------------------------
def cmd_online(args):
    from .evaluation.online_optimizer import oracle_experiment, subspace_optimizer_experiment
    cfg = load_config(args.config)
    exp = _exp_dir(cfg)
    seed = int(cfg.experiment.get("seeds", [0])[0])
    n_steps = int(cfg.get("online", {}).get("steps", 400)) if hasattr(cfg, "get") else 400
    oracle = oracle_experiment(cfg, seed, n_steps=n_steps, device=args.device)
    sub = subspace_optimizer_experiment(cfg, seed, n_steps=n_steps, device=args.device)
    # drop bulky per-record list before saving summary
    summary = {k: v for k, v in oracle.items() if k != "records"}
    (exp / "oracle_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (exp / "subspace_optimizer.json").write_text(json.dumps(sub, indent=2), encoding="utf-8")
    save_run_metadata(exp, cfg, extra={"command": "run-online-optimizer"})
    print(f"[online] oracle predicted_win_rate={oracle['predicted_win_rate']:.3f} "
          f"mean_improvement={oracle['mean_improvement_over_baseline']:.3e}")


# -- report -------------------------------------------------------------------
def cmd_report(args):
    from .evaluation.reports import make_report
    p = make_report(args.experiment_dir)
    print(f"[make-report] wrote {p}")


# -- end-to-end convenience ---------------------------------------------------
def cmd_run_e1(args):
    cmd_generate(args)
    cmd_build_dataset(args)
    cmd_train_predictor(args)
    cmd_online(args)
    cfg = load_config(args.config)
    cmd_report(argparse.Namespace(experiment_dir=str(_exp_dir(cfg))))


def build_parser():
    p = argparse.ArgumentParser(prog="tad", description="Topology-Aware Dynamics")
    p.add_argument("--device", default="cpu")
    sub = p.add_subparsers(dest="command", required=True)

    def add(name, fn, **kw):
        sp = sub.add_parser(name)
        sp.set_defaults(func=fn)
        return sp

    sp = add("generate-trajectories", cmd_generate); sp.add_argument("--config", required=True)
    sp = add("build-dataset", cmd_build_dataset); sp.add_argument("--config", required=True)
    sp = add("train-predictor", cmd_train_predictor); sp.add_argument("--config", required=True)
    sp = add("evaluate-predictor", cmd_evaluate_predictor)
    sp.add_argument("--config", required=True); sp.add_argument("--checkpoint", default=None)
    sp = add("analyze-temporal-structure", cmd_analyze); sp.add_argument("--run-dir", required=True)
    sp = add("run-online-optimizer", cmd_online); sp.add_argument("--config", required=True)
    sp = add("make-report", cmd_report); sp.add_argument("--experiment-dir", required=True)
    sp = add("run-e1", cmd_run_e1); sp.add_argument("--config", required=True)
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    # propagate top-level --device default if subparser lacks it
    if not hasattr(args, "device"):
        args.device = "cpu"
    args.func(args)


if __name__ == "__main__":
    main()
