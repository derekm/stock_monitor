#!/usr/bin/env python3
"""
regime_serving.py — serve regime-selected Granite-TTM checkpoints in
production.

Bridges pass6/pass7 research (per-regime fine-tuned models, validated OOS)
into forecast_granite's daily serving path. For each ticker:

  1. Read regime_model_best.csv for the CURRENT HMM regime (the pass6
     best-config selection: max OOS direction excess over the regime's
     persistence baseline).
  2. If a checkpoint exists for (ticker, regime) under checkpoints/regime/,
     return it — the forecast is regime-SELECTED.
  3. Otherwise return None — the caller uses the general model, and the
     flag says why (no regime model / no checkpoint).

The selection is per (ticker, current regime). A ticker with no pass6
coverage simply keeps using the general model — this is the honest
degradation: regime selection is an upgrade when available, never a
downgrade.

Output: (checkpoint_path | None, selection_dict | None, reason)
"""
from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd

from granite_config import DEFAULT_MODEL  # noqa: F401  (canonical model id)

DATA_DIR = Path(__file__).parent
REGIME_BEST = DATA_DIR / "regime_model_best.csv"
HMM_FILE = DATA_DIR / "hmm_regime_states.parquet"
CKPT_DIR = DATA_DIR / "checkpoints" / "regime"


def current_regime() -> str | None:
    """Latest HMM regime label, or None when unavailable."""
    if not HMM_FILE.exists():
        return None
    try:
        hmm = pd.read_parquet(HMM_FILE)
        if "date" in hmm.columns and "regime" in hmm.columns:
            hmm["date"] = pd.to_datetime(hmm["date"], errors="coerce")
            hmm = hmm.dropna(subset=["date"]).sort_values("date")
            if len(hmm):
                return str(hmm.iloc[-1]["regime"])
    except Exception:
        pass
    return None


def best_config_for(ticker: str, regime: str | None) -> dict | None:
    """pass6 best config (steps/cap/lr/dir/pers) for (ticker, regime)."""
    if not REGIME_BEST.exists() or not regime:
        return None
    try:
        rb = pd.read_csv(REGIME_BEST)
        row = rb[(rb["ticker"].astype(str).str.upper() == ticker.upper())
                 & (rb["regime"].astype(str) == regime)]
        if not len(row):
            return None
        r = row.iloc[0]
        cfg = {
            "steps": int(r["steps"]) if pd.notna(r.get("steps")) else None,
            "cap": int(r["cap"]) if pd.notna(r.get("cap")) else None,
            "lr": r.get("lr"),
            "dir_acc": float(r["dir_acc"]) if pd.notna(r.get("dir_acc")) else None,
            "pers_dir": float(r["pers_dir"]) if pd.notna(r.get("pers_dir")) else None,
        }
        # per-span direction accuracy (production horizons 10..96)
        for s in (10, 21, 42, 63, 96):
            col = f"dir_acc_h{s}"
            if col in rb.columns and pd.notna(r.get(col)):
                cfg[f"dir_acc_h{s}"] = float(r[col])
        return cfg
    except Exception:
        return None


def serve_regime_model(ticker: str) -> tuple[Path | None, dict | None, str]:
    """(ckpt_path, best_config, reason) for serving (ticker) today.

    reason ∈ {"served", "no_regime", "no_coverage", "no_checkpoint"}.
    """
    reg = current_regime()
    if not reg:
        return None, None, "no_regime"
    cfg = best_config_for(ticker, reg)
    if cfg is None:
        return None, None, "no_coverage"
    fname = f"{ticker.upper()}__{reg}__{cfg['steps']}__{_lr_safe(cfg['lr'])}.pt"
    path = CKPT_DIR / fname
    if path.exists():
        cfg["n_channels"] = _ckpt_channels(path)
        cfg["trained_on"] = _ckpt_trained_on(path)
        cfg["age_days"] = (pd.Timestamp.today() - cfg["trained_on"]).days \
            if cfg["trained_on"] is not None else None
        return path, cfg, "served"
    return None, cfg, "no_checkpoint"


def _ckpt_trained_on(path: Path):
    try:
        state = __import__("torch").load(path, map_location="cpu")
        ts = state.get("trained_on")
        if ts is not None:
            return pd.Timestamp(ts)
        import os
        return pd.Timestamp.fromtimestamp(os.path.getmtime(path))
    except Exception:
        return None


def _ckpt_channels(path: Path) -> int:
    try:
        state = __import__("torch").load(path, map_location="cpu")
        return int(state.get("n_channels", 1))
    except Exception:
        return 1


def _lr_safe(lr) -> str:
    return str(lr).replace(".", "p") if lr is not None else "None"


def load_regime_model(model_cls, ckpt_path: Path, base_model):
    """Instantiate a fresh copy of the base and load the regime state dict.

    Returns the model (copy of base + regime weights). The caller must
    .eval() it and move it to the right device.
    """
    m = copy.deepcopy(base_model)
    state = __import__("torch").load(ckpt_path, map_location="cpu")
    m.load_state_dict(state["model"])
    return m


def serving_report(tickers: list[str]) -> pd.DataFrame:
    """Table of what regime model each ticker would get served today."""
    reg = current_regime()
    rows = []
    for t in tickers:
        path, cfg, reason = serve_regime_model(t)
        rows.append({
            "ticker": t, "regime": reg, "reason": reason,
            "steps": cfg["steps"] if cfg else None,
            "dir_acc": cfg["dir_acc"] if cfg else None,
            "excess": (cfg["dir_acc"] - cfg["pers_dir"]) if cfg and cfg["dir_acc"] is not None and cfg["pers_dir"] is not None else None,
            "ckpt": str(path) if path else "",
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # quick report mode
    import sys
    tickers = [t.strip().upper() for t in (sys.argv[1] or "AEP,NVR,FICO").split(",") if t.strip()]
    print(serving_report(tickers).to_string(index=False))
