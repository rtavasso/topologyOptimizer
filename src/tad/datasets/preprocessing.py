"""Build the supervised dynamics dataset across runs (Sections 9, 15.1).

Splits strictly by whole run/trajectory (never random windows), keeping teacher
matrices and seeds disjoint across splits to prevent leakage (Section 20.8).
Produces one ``.npz`` per (target, history_length, horizon).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..logging.reader import TrajectoryReader
from .trajectory_windows import build_run_windows, WindowTensors


def split_runs(run_dirs: List[Path], fractions=(0.7, 0.15, 0.15), seed: int = 0) -> Dict[str, List[Path]]:
    """Deterministic split by whole run. Returns train/val/test path lists."""
    run_dirs = sorted(run_dirs, key=lambda p: str(p))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(run_dirs))
    n = len(run_dirs)
    n_tr = max(1, int(round(fractions[0] * n)))
    n_va = max(0, int(round(fractions[1] * n)))
    if n_tr + n_va >= n and n >= 3:
        n_tr, n_va = n - 2, 1
    tr = [run_dirs[i] for i in order[:n_tr]]
    va = [run_dirs[i] for i in order[n_tr:n_tr + n_va]]
    te = [run_dirs[i] for i in order[n_tr + n_va:]]
    return {"train": tr, "val": va or te[:1], "test": te or va[:1]}


def _gather(run_dirs, hl, h, target, reps, node_feat_dim, run_offset, stride=1) -> Optional[WindowTensors]:
    parts = []
    shapes, mask, names = None, None, None
    # compute global max target dim first for consistent padding
    max_dim = 0
    readers = [TrajectoryReader(rd) for rd in run_dirs]
    for r in readers:
        from .trajectory_windows import TARGET_BUILDERS, DELTA_TARGETS
        fn = TARGET_BUILDERS.get("future_weight" if target in DELTA_TARGETS else target)
        for nm in r.layer_names:
            s = fn(r, nm, 0)
            max_dim = max(max_dim, int(np.prod(s.shape)))
    for k, r in enumerate(readers):
        w = build_run_windows(r, hl, h, target, reps, node_feat_dim=node_feat_dim,
                              sketch_seed=0, run_index=run_offset + k, max_target_dim=max_dim,
                              stride=stride)
        if w is None:
            continue
        parts.append(w)
        shapes, mask, names = w.node_shapes, w.mask, r.layer_names
    if not parts:
        return None
    cat = lambda attr: np.concatenate([getattr(p, attr) for p in parts], axis=0)
    meta = {k: np.concatenate([p.meta[k] for p in parts]) for k in parts[0].meta}
    aux = None
    if parts[0].aux is not None:
        aux = {
            "aux_W": np.concatenate([p.aux["aux_W"] for p in parts], axis=0),
            "aux_GM_hist": np.concatenate([p.aux["aux_GM_hist"] for p in parts], axis=0),
            "aux_W_shapes": parts[0].aux["aux_W_shapes"],
            "aux_GM_shape": parts[0].aux["aux_GM_shape"],
        }
    return WindowTensors(cat("feat_hist"), cat("tgt_hist"), cat("Y"),
                         mask, shapes, meta, aux=aux), names


def build_dynamics_dataset(cfg, trajectory_dir, out_dir) -> dict:
    trajectory_dir = Path(trajectory_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dirs = [p for p in sorted(trajectory_dir.iterdir())
                if p.is_dir() and (p / "tensors.zarr").exists()]
    if not run_dirs:
        raise FileNotFoundError(f"no runs with trajectories under {trajectory_dir}")

    dd = cfg.dynamics_dataset
    history_lengths = list(dd.get("history_lengths", [1, 4, 16, 32]))
    horizons = list(dd.get("horizons", [1, 2, 4, 8, 16, 32, 64, 100]))
    targets = list(dd.get("targets", ["future_gradient", "weight_delta", "probe_action"]))
    reps = list(dd.get("representations", ["R1", "R2", "R3", "R5"]))
    node_feat_dim = int(dd.get("node_feat_dim", 128))
    stride = int(dd.get("window_stride", 1))
    split = split_runs(run_dirs, seed=int(cfg.get("experiment", {}).get("split_seed", 0))
                       if hasattr(cfg, "get") else 0)

    # verify no leakage of seeds/run ids across splits
    _assert_no_leakage(split)

    manifest = {"targets": targets, "history_lengths": history_lengths, "horizons": horizons,
                "representations": reps, "window_stride": stride,
                "splits": {k: [p.name for p in v] for k, v in split.items()}}

    offsets = {"train": 0, "val": 1000, "test": 2000}
    n_combos = len(targets) * len(history_lengths) * len(horizons)
    print(f"[build-dataset] {len(run_dirs)} runs -> {n_combos} (target,H,horizon) datasets "
          f"(window_stride={stride})", flush=True)
    done = 0
    for target in targets:
        for hl in history_lengths:
            for h in horizons:
                bundle = {}
                names = None
                for sp in ("train", "val", "test"):
                    res = _gather(split[sp], hl, h, target, reps, node_feat_dim, offsets[sp], stride=stride)
                    if res is None:
                        continue
                    w, names = res
                    bundle[f"{sp}_feat"] = w.feat_hist
                    bundle[f"{sp}_tgt_hist"] = w.tgt_hist
                    bundle[f"{sp}_Y"] = w.Y
                    for mk, mv in w.meta.items():
                        bundle[f"{sp}_meta_{mk}"] = mv
                    bundle["mask"] = w.mask
                    bundle["node_shapes"] = np.array(w.node_shapes)
                    bundle["cadence"] = int(w.meta["cadence"][0])
                    bundle["h_idx"] = int(w.meta["h_idx"][0])
                    bundle["target"] = target
                    if w.aux is not None:
                        for ak, av in w.aux.items():
                            bundle[f"{sp}_{ak}"] = av
                if not bundle:
                    continue
                bundle["layer_names"] = np.array(names if names else [])
                fname = f"{target}__H{hl}__h{h}.npz"
                np.savez_compressed(out_dir / fname, **bundle)
                done += 1
                n_train = bundle.get("train_Y", np.empty((0,))).shape[0]
                print(f"[build-dataset] [{done}/{n_combos}] {target} H{hl} h{h} "
                      f"({n_train} train windows)", flush=True)

    with open(out_dir / "dataset_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def load_dynamics_dataset(path) -> dict:
    return dict(np.load(path, allow_pickle=True))


def _run_identity(run_dir: Path) -> dict:
    """Extract (seed, teacher fingerprint) for leakage checks (spec 20.8)."""
    ident = {"seed": None, "teacher": None}
    meta_path = run_dir / "run_metadata.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            ident["seed"] = meta.get("seed")
            ident["teacher"] = meta.get("teacher_fingerprint")
        except Exception:
            pass
    if ident["seed"] is None and "seed" in run_dir.name:
        try:
            ident["seed"] = int(run_dir.name.split("seed")[-1])
        except ValueError:
            pass
    return ident


def _assert_no_leakage(split: Dict[str, List[Path]]) -> None:
    """No run_id, seed, or teacher may appear in more than one split (spec 20.8)."""
    seen_run: Dict[str, str] = {}
    seen_seed: Dict[object, str] = {}
    seen_teacher: Dict[object, str] = {}
    for sp, dirs in split.items():
        for d in dirs:
            if d.name in seen_run and seen_run[d.name] != sp:
                raise AssertionError(f"run {d.name} appears in both {seen_run[d.name]} and {sp}")
            seen_run[d.name] = sp
            ident = _run_identity(d)
            for key, store, label in ((ident["seed"], seen_seed, "seed"),
                                      (ident["teacher"], seen_teacher, "teacher")):
                if key is None:
                    continue
                if key in store and store[key] != sp:
                    raise AssertionError(
                        f"{label} {key!r} appears in both {store[key]} and {sp} (train/test leakage)")
                store[key] = sp
