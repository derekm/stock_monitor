#!/usr/bin/env python3
"""
ttm_backfill.py — Config + callback driven library for pre-training Granite
TinyTimeMixer (TTM) over the full daily_price history for an arbitrary set of
tickers, at arbitrary model regimes (global aggregate, proxy-padded aggregate,
and per-ticker), with arbitrary data sources (adjusted / unadjusted closes).

This is the factored-out core that granite_backfill.py used to inline. It is
designed to be driven by *configs* and *callbacks* so that arbitrary model
regimes can be composed at the global and per-ticker level without editing the
training loop — mirroring how fisher_index.py generalizes index construction.

--------------------------------------------------------------------------------
CORE CONCEPTS
--------------------------------------------------------------------------------
* ``DataConfig``   — what data to use (price column, cleaning, which tickers,
                     context/horizon, window sampling, short-ticker padding).
* ``TrainConfig``  — how to train one checkpoint (steps, batch, lr, optimizer,
                     grad-clip, dtype, warm-start source).
* ``RegimeConfig`` — a named training *regime*: a TrainConfig + which tickers it
                     applies to (full / short / all) + where to save + which
                     prior regime to warm-start from.
* ``BackfillConfig`` — a full job: a DataConfig + an ordered list of regimes +
                     an optional comparison step + callbacks.
* ``Callbacks``    — hooks fired during the run (window build, per-train-step,
                     per-ticker, per-regime, comparison) so callers can observe
                     / log / abort without touching the engine.

The default ``default_backfill_config()`` reproduces the historical
granite_backfill.run() behavior exactly (unadjusted closes, 3 regimes: global
aggregate -> padded aggregate -> per-ticker). Adjusted backfills are produced by
``adjusted_backfill_config()`` (same regime structure, adj_close, excluded
no-adj tickers, separate output dirs) — the ONLY difference is the price source,
so the two are directly comparable.

A "regime" at the per-ticker level means: a per-ticker checkpoint family trained
with a given TrainConfig. Multiple regimes can share the same global/padded base
but differ in lr / steps / optimizer, enabling the pass-3/4 parameter sweeps to
be expressed as config rather than copy-pasted loops.

--------------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------------
    from ttm_backfill import default_backfill_config, run_backfill
    cfg = default_backfill_config(steps=150, batch=16)
    run_backfill(cfg)

    # adjusted variant — identical recipe, different price source:
    cfg = adjusted_backfill_config(steps=150, batch=16)
    run_backfill(cfg)

    # custom regime sweep at per-ticker level:
    cfg = default_backfill_config()
    cfg.regimes.append(RegimeConfig(
        name="per_ticker_lr3e-4", applies_to="full", out_dir=cfg.data.out_dir/"per_ticker_lr3e-4",
        warm_from="per_ticker", train=TrainConfig(steps=150, lr=3e-4)))
    run_backfill(cfg)
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import polars as pl

import granite_daily as gd
import window_padding as wp

# device selection lives in granite_daily; alias for this module
_device = gd._device

HERE = Path(__file__).parent
PRICES = HERE / "daily_prices.parquet"
CKPT_DIR = gd.CKPT_DIR
GLOBAL_DIR = gd.GLOBAL_DIR
PADDED_DIR = gd.PADDED_DIR
PER_TICKER_DIR = gd.PER_TICKER_DIR

# sensible library defaults (override via config)
CONTEXT = gd.CONTEXT        # 512 (TTM-r2 has a FIXED context; never shrunk)
HORIZON = gd.HORIZON        # 96
MIN_NEEDED = HORIZON + 1    # need at least 1 real close to pad the rest


# =============================================================================
# CONFIG DATACLASSES
# =============================================================================
@dataclass
class DataConfig:
    """What data the job trains on and how windows are sampled."""
    use_adj: bool = False                       # True -> train on adj_close
    recent_trading_days: Optional[int] = None   # None = full history
    max_daily_logret: float = 0.3               # drop impossible single-day moves
    context: int = CONTEXT
    horizon: int = HORIZON
    max_windows_per_ticker: int = 200           # linspace cap (matches old default)
    # short-ticker padding
    pad_short: bool = True
    exclude_tickers: list[str] = field(default_factory=list)   # e.g. no-adj tickers
    tickers: Optional[list[str]] = None         # None = all (after exclusions)
    out_dir: Path = CKPT_DIR


@dataclass
class TrainConfig:
    """How to train ONE checkpoint (global / padded / per-ticker)."""
    steps: int = 150
    batch: int = gd.BATCH
    lr: float = gd.LR
    optimizer: str = "adamw"
    grad_clip: float = 5.0
    dtype: str = "float32"
    compare: bool = False


@dataclass
class RegimeConfig:
    """A named training regime.

    `applies_to` controls which tickers this regime trains:
        "full"  -> tickers with >= context real closes
        "short" -> tickers with < context (proxy-padded)
        "all"   -> both (used for the global aggregate)
    `warm_from` names a prior regime to warm-start from:
        "pretrained" -> IBM zero-shot base
        "global"     -> the global aggregate regime
        "padded"     -> the padded aggregate regime
        "self"       -> this regime's own latest ckpt (incremental)
        None         -> pretrained
    """
    name: str
    applies_to: str = "full"            # full | short | all
    out_dir: Path = PER_TICKER_DIR
    warm_from: Optional[str] = "pretrained"
    train: TrainConfig = field(default_factory=TrainConfig)
    kind: str = "per_ticker"            # per_ticker | aggregate
    # When set, this regime uses its OWN window config (context/horizon/source)
    # instead of the job-wide DataConfig. This makes context length, horizon,
    # and price source first-class per-regime parameters (the pass-3/4 dims).
    data: Optional[DataConfig] = None
    # When set, the model is BUILT (not loaded from the default IBM ckpt) with
    # these TTM hyperparameters: {"context_length", "prediction_length",
    # "patch_length", "use_decoder"}. Horizon/context are model-architecture
    # params in TTM (head shape depends on horizon), so a horizon sweep MUST
    # construct a fresh model here rather than only changing window shape.
    model_config: Optional[dict] = None
    # For aggregate regimes, sample per-ticker windows down to this cap so the
    # stacked tensor fits in RAM (diversity seed, not exhaustive).
    agg_windows_per_ticker: int = 200


@dataclass
class Callbacks:
    """Hooks fired during the run. All optional. Return value of on_train_step
    may be ignored; on_ticker / on_regime may return False to abort the run."""
    on_window_build: Optional[Callable[[int, float], None]] = None   # (n_tickers, secs)
    on_train_step: Optional[Callable[[dict], None]] = None          # step telemetry
    on_train_end: Optional[Callable[[dict], None]] = None           # {name, steps, n, secs, out_path}
    on_ticker: Optional[Callable[[dict], None]] = None              # {i, n, tk, wins, secs}
    on_regime: Optional[Callable[[dict], None]] = None              # {regime, phase}
    on_compare: Optional[Callable[[list[dict]], None]] = None       # comparison rows
    on_log: Optional[Callable[[str], None]] = None                 # arbitrary message


@dataclass
class BackfillConfig:
    data: DataConfig = field(default_factory=DataConfig)
    regimes: list[RegimeConfig] = field(default_factory=list)
    compare: bool = True
    callbacks: Callbacks = field(default_factory=Callbacks)
    # comparison output
    compare_log: Path = HERE / "ttm_backfill_compare.jsonl"


# =============================================================================
# FACTORY DEFAULTS
# =============================================================================
def _regime(name, applies_to, out_dir, warm_from, train, kind="per_ticker", agg=200):
    return RegimeConfig(name=name, applies_to=applies_to, out_dir=out_dir,
                        warm_from=warm_from, train=train, kind=kind,
                        agg_windows_per_ticker=agg)


def default_backfill_config(steps: int = 150, batch: int | None = None,
                            lr: float = gd.LR, use_adj: bool = False,
                            exclude_tickers: Optional[list[str]] = None,
                            tickers: Optional[list[str]] = None) -> BackfillConfig:
    """Reproduce granite_backfill.run(): global -> padded -> per-ticker, all on
    the same data source. warm chain: global<-pretrained, padded<-global,
    per_ticker<-global (or padded for shorts)."""
    b = batch or gd.BATCH
    tc = lambda s=steps, l=lr, bt=b: TrainConfig(steps=s, batch=bt, lr=l)
    regimes = [
        _regime("global", "all", GLOBAL_DIR, "pretrained", tc(), kind="aggregate", agg=200),
        _regime("padded", "short", PADDED_DIR, "global", tc(), kind="aggregate", agg=200),
        _regime("per_ticker", "full", PER_TICKER_DIR, "global", tc()),
    ]
    # shorts warm from padded, not global
    regimes[2] = replace(regimes[2], warm_from="global")  # full -> global
    data = DataConfig(use_adj=use_adj, max_windows_per_ticker=200,
                      exclude_tickers=list(exclude_tickers or []),
                      tickers=tickers, out_dir=CKPT_DIR)
    cfg = BackfillConfig(data=data, regimes=regimes, compare=True)
    # mark shorts to warm from padded via a per-ticker override map
    cfg._short_warm = "padded"
    return cfg


def no_adj_tickers(prices: pd.DataFrame) -> list[str]:
    """Tickers whose adj_close is essentially identical to close (no adjustment
    data) — must be excluded from adjusted training so they don't train on a
    flat (unadjusted) series masquerading as adjusted."""
    out = set()
    for tk, sub in prices.groupby("ticker"):
        if "adj_close" not in sub.columns or sub["adj_close"].isna().all():
            out.add(tk); continue
        c = sub["close"]; a = sub["adj_close"]
        m = (c > 0) & (a > 0) & c.notna() & a.notna()
        if m.sum() == 0:
            out.add(tk); continue
        rel = (a[m] - c[m]).abs() / c[m].abs()
        if rel.mean() < 1e-4:
            out.add(tk)
    return sorted(out)


def adjusted_backfill_config(steps: int = 150, batch: int | None = None,
                             lr: float = gd.LR,
                             prices: Optional[pd.DataFrame] = None) -> BackfillConfig:
    """Identical recipe to default_backfill_config but on adj_close, excluding
    tickers with no adjusted history. The ONLY variable vs the unadjusted run
    is the price source -> directly comparable checkpoints."""
    df = prices if prices is not None else pd.read_parquet(PRICES)
    exclude = no_adj_tickers(df)
    cfg = default_backfill_config(steps=steps, batch=batch, lr=lr, use_adj=True,
                                  exclude_tickers=exclude)
    ADJ = CKPT_DIR
    cfg.regimes[0] = replace(cfg.regimes[0], out_dir=ADJ / "adjusted_global")
    cfg.regimes[1] = replace(cfg.regimes[1], out_dir=ADJ / "adjusted_padded")
    cfg.regimes[2] = replace(cfg.regimes[2], out_dir=ADJ / "adjusted_per_ticker")
    cfg.data = replace(cfg.data, out_dir=ADJ)
    return cfg


# =============================================================================
# DATA PREP (reuse granite_backfill._clean_price_frame for identical hygiene)
# =============================================================================
def _clean_price_frame(prices: pd.DataFrame, recent_trading_days: int | None = None,
                       use_adj: bool = True, max_daily_logret: float = 0.3) -> pd.DataFrame:
    """Normalize daily_prices before windowing.

    The raw parquet has several defects that corrupt TTM training:
      1) Merged duplicate pulls -> same (ticker, date) appears multiple times,
         sometimes with CONFLICTING close (two yfinance pulls disagree).
      2) Unadjusted long history -> pre-split prices span ~100x scale. Now that
         adj_close is captured, the adjusted series is stationary across decades,
         so we train on adj_close.
      3) Split/dividend adjustment ERRORS inject impossible single-day moves
         (e.g. AEP 131.7 -> 33.7 -> 131.4 in one day). Drop rows whose adjacent
         adj_close log-return exceeds `max_daily_logret`.

    Fixes, in order: (a) drop exact dup rows, (b) clip to recent days,
    (c) drop impossible daily moves, (d) collapse (ticker,date) conflicts by mean.
    The training price is returned in a `close` column so downstream is agnostic.
    """
    price_col = "adj_close" if (use_adj and "adj_close" in prices.columns) else "close"
    if price_col not in prices.columns:
        raise ValueError(f"_clean_price_frame: missing column {price_col!r}")
    need = ["ticker", "date", price_col]
    df = prices.drop_duplicates(subset=need).copy()
    if recent_trading_days is not None and recent_trading_days > 0:
        df = (
            df.sort_values(["ticker", "date"])
            .groupby("ticker", group_keys=False)
            .tail(recent_trading_days)
        )
    if max_daily_logret and max_daily_logret > 0:
        df = df.sort_values(["ticker", "date"])
        g = df.groupby("ticker", group_keys=False)
        logret = g[price_col].transform(lambda s: np.log(s / s.shift(1)))
        fwd = logret
        bwd = -g[price_col].transform(lambda s: np.log(s / s.shift(-1)))
        drop_mask = (fwd.abs() > max_daily_logret) | (bwd.abs() > max_daily_logret)
        n_bad = int(drop_mask.sum())
        if n_bad:
            print(f"  [_clean] dropping {n_bad} impossible-move rows "
                  f"(|adj logret|>{max_daily_logret})", flush=True)
        df = df[~drop_mask]
    df = df.drop(columns=[c for c in ("close", "adj_close") if c in df.columns and c != price_col])
    before = len(df)
    df = (
        df.groupby(["ticker", "date"], as_index=False)
        .agg({price_col: "mean", **{c: "first" for c in df.columns if c not in ("ticker", "date", price_col)}})
    )
    df = df.rename(columns={price_col: "close"})
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    clipmsg = f"clip{recent_trading_days}d" if recent_trading_days else "full-history"
    print(f"[_clean_price_frame] rows {before} -> {len(df)} "
          f"(used {price_col}; dedup+{clipmsg}+drop-spikes+conflict-mean)", flush=True)
    return df


def per_ticker_plan(prices: pd.DataFrame):
    """Return list of (ticker, n_real, padded) where padded=True means the ticker
    has < CONTEXT real closes and will be window-filled with a sector/market proxy
    head. Built with a single polars group_by (multithreaded)."""
    plan = []
    cons = gd._constituents_df()
    sector_of = {}
    if cons is not None and not cons.empty and "gics_sector" in cons.columns:
        sector_of = dict(zip(cons["ticker"], cons["gics_sector"]))
    plf = pl.from_pandas(prices[["ticker", "date", "close"]])
    counts = {
        (tk[0] if isinstance(tk, tuple) else tk): int(n)
        for tk, n in plf.group_by("ticker", maintain_order=True)
        .len().select("ticker", "len").iter_rows()
    }
    for tk, n in counts.items():
        sector = sector_of.get(tk)
        plan.append((tk, n, n >= CONTEXT, sector))
    return plan


def build_windows(cfg: DataConfig, prices: pd.DataFrame) -> dict[str, list[tuple]]:
    """Build (context, target) windows per ticker, honoring DataConfig.

    Mirrors the old build_full_history_windows: full tickers -> raw rolling
    512-windows (stride-1, linspace-cap at max_windows_per_ticker); short tickers
    -> proxy-padded context head + own target. Returns {ticker: [wins]}."""
    cons = gd._constituents_df()
    tk_set = set(t.upper() for t in cfg.tickers) if cfg.tickers else None
    exclude = {t.upper() for t in cfg.exclude_tickers}

    prices = _clean_price_frame(prices, recent_trading_days=cfg.recent_trading_days,
                                use_adj=cfg.use_adj,
                                max_daily_logret=cfg.max_daily_logret)
    if exclude:
        prices = prices[~prices["ticker"].isin(exclude)].copy()

    C, H = cfg.context, cfg.horizon
    plf = pl.from_pandas(prices[["ticker", "date", "close"]]).sort(["ticker", "date"])
    by_ticker = {
        (tk[0] if isinstance(tk, tuple) else tk): s["close"].to_numpy().astype(np.float32)
        for tk, s in plf.group_by("ticker", maintain_order=True)
    }
    all_dates = np.sort(plf["date"].unique().to_numpy())
    last_date_by_ticker = {
        (tk[0] if isinstance(tk, tuple) else tk): d
        for tk, d in plf.group_by("ticker", maintain_order=True).agg(pl.col("date").max()).iter_rows()
    }
    sector_of = {}
    if cons is not None and not cons.empty and "gics_sector" in cons.columns:
        sector_of = dict(zip(cons["ticker"], cons["gics_sector"]))

    proxy_cache = {}
    wins_by_ticker: dict[str, list] = {}
    for tk, s in by_ticker.items():
        if tk_set is not None and tk not in tk_set:
            continue
        sector = sector_of.get(tk)
        if len(s) < MIN_NEEDED:
            continue
        if len(s) >= C:
            n_windows = len(s) - (C + H) + 1
            idxs = np.arange(n_windows)
            if len(idxs) > cfg.max_windows_per_ticker:
                idxs = np.linspace(0, n_windows - 1, cfg.max_windows_per_ticker).astype(int)
            for k in idxs:
                c = s[k: k + C]
                tgt = s[k + C: k + C + H]
                if len(c) == C and len(tgt) == H:
                    wins_by_ticker.setdefault(tk, []).append((c, tgt, tk))
        elif cfg.pad_short:
            idxs = range(H, len(s) + 1)
            if len(list(idxs)) > cfg.max_windows_per_ticker:
                idxs = np.linspace(H, len(s), cfg.max_windows_per_ticker).astype(int)
            for e in idxs:
                ctx_full = wp.pad_to_context(tk, s[:e].astype(np.float32),
                                             sector=sector, px=prices, cons=cons,
                                             proxy_cache=proxy_cache,
                                             last_date=last_date_by_ticker.get(tk),
                                             all_dates=all_dates)
                tgt = s[e - H: e]
                if len(tgt) == H and len(ctx_full) == C:
                    wins_by_ticker.setdefault(tk, []).append((ctx_full, tgt, tk))
    return wins_by_ticker


# =============================================================================
# TRAIN / SCORE (refactored; identical numerics to old train_windows/aggregate)
# =============================================================================
def make_model(context_length: int = CONTEXT, prediction_length: int = HORIZON,
               patch_length: int = 64, use_decoder: bool = True,
               device=None, warm_sd: Optional[dict] = None):
    """Construct a TTM model with arbitrary context/horizon/patch/decoder.

    context_length and prediction_length are MODEL hyperparameters in TTM (the
    output head shape depends on horizon), so horizon/context sweeps must build
    a fresh model via TinyTimeMixerConfig — not merely change the window shape.
    When `warm_sd` is given it is loaded with strict=False (shapes may differ);
    otherwise the IBM pretrained weights seed the new architecture.
    """
    from tsfm_public.models.tinytimemixer import TinyTimeMixerConfig, TinyTimeMixerForPrediction
    cfg = TinyTimeMixerConfig.from_pretrained(gd.DEFAULT_MODEL)
    cfg.context_length = context_length
    cfg.prediction_length = prediction_length
    cfg.patch_length = patch_length
    cfg.use_decoder = use_decoder
    m = TinyTimeMixerForPrediction(cfg)
    if warm_sd is not None:
        try:
            m.load_state_dict(warm_sd, strict=False)
        except Exception as e:
            print(f"    [warm partial {context_length}/{prediction_length}]: {e}", flush=True)
    if device is not None:
        m = m.to(device)
    return m


def _make_optimizer(model, tc: TrainConfig):
    if tc.optimizer == "adamw":
        import torch
        return torch.optim.AdamW(model.parameters(), lr=tc.lr)
    if tc.optimizer == "adam":
        import torch
        return torch.optim.Adam(model.parameters(), lr=tc.lr)
    if tc.optimizer == "sgd":
        import torch
        return torch.optim.SGD(model.parameters(), lr=tc.lr, momentum=0.9)
    raise ValueError(f"unknown optimizer {tc.optimizer!r}")


def _train_loop(model, dl, tc: TrainConfig, device, cb: Callbacks, meta: dict):
    import torch
    model.train()
    opt = _make_optimizer(model, tc)
    t0 = time.time()
    step = 0
    while step < tc.steps:
        for xb, yb in dl:
            xb = xb.to(device, non_blocking=True).to(getattr(torch, tc.dtype))
            yb = yb.to(device, non_blocking=True).to(getattr(torch, tc.dtype))
            out = model(past_values=xb, future_values=yb)
            loss = out.loss
            if not torch.isfinite(loss):
                print(f"    [NaN/inf loss at step {step}; aborting]", flush=True)
                return step, t0
            opt.zero_grad()
            loss.backward()
            if tc.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
            opt.step()
            step += 1
            if cb.on_train_step:
                cb.on_train_step({"step": step, "loss": float(loss), **meta})
            if step >= tc.steps:
                break
    return step, t0


def train_checkpoint(wins, tc: TrainConfig, device, out_dir: Path, name: str,
                     warm_sd: Optional[dict] = None, model=None, base_model=None,
                     cb: Callbacks = None, meta: dict = None,
                     model_config: Optional[dict] = None) -> Optional[Path]:
    """Train a single checkpoint. Reuses a loaded `model`/`warm_sd` when given
    (per-ticker path) to avoid re-loading the heavy base on every call. When
    `model_config` is given, a FRESH model is built with those TTM hyperparameters
    (context/horizon/patch/decoder) and `warm_sd` is applied with strict=False —
    required for horizon/context sweeps where the head shape differs from base.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    cb = cb or Callbacks()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not wins:
        return None
    ctx = np.stack([w[0] for w in wins])[:, :, None]
    tgt = np.stack([w[1] for w in wins])[:, :, None]
    ds = TensorDataset(torch.tensor(ctx), torch.tensor(tgt))
    dl = DataLoader(ds, batch_size=tc.batch, shuffle=True, pin_memory=True)

    if model_config is not None:
        # build a fresh model with the regime's architecture; partial warm-start
        model = make_model(device=device, warm_sd=warm_sd, **model_config)
    elif model is None:
        model, _ = gd.load_granite_model(gd.DEFAULT_MODEL)
        model = model.to(device)
    if warm_sd is not None and model_config is None:
        try:
            model.load_state_dict(warm_sd)
        except Exception as e:
            print(f"  warm-start failed ({e}); using pretrained")
    n = len(wins)
    meta = {**(meta or {}), "name": name, "n": n, "out_dir": str(out_dir)}
    step, t0 = _train_loop(model, dl, tc, device, cb, meta)
    d = date.today().isoformat().replace("-", "")
    out_path = out_dir / f"{name}_{d}.pt"
    # Only move an internally-created model to CPU before saving (memory). A
    # passed-in model (the shared per-ticker base) must stay on its device so
    # the caller can reuse it for the next ticker.
    if model is None and device.type == "cuda":
        model = model.cpu()
    torch.save(model.state_dict(), out_path)
    model.eval()
    print(f"  trained {step} steps on {n} windows; ckpt={out_path.name} "
          f"({time.time()-t0:.1f}s, device={device.type})", flush=True)
    if cb.on_train_end:
        cb.on_train_end({"name": name, "steps": step, "n": n,
                         "secs": time.time() - t0, "out_path": str(out_path)})
    return out_path


def score_windows(model, wins, device) -> float:
    """Mean absolute error of `model` over `wins`. Identical to old score_windows."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    if not wins:
        return float("nan")
    ctx = np.stack([w[0] for w in wins])[:, :, None]
    tgt = np.stack([w[1] for w in wins])[:, :, None]
    ds = TensorDataset(torch.tensor(ctx), torch.tensor(tgt))
    dl = DataLoader(ds, batch_size=gd.BATCH, shuffle=False)
    model.eval()
    errs = []
    with torch.no_grad():
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            out = model(past_values=xb)
            p = getattr(out, "prediction_outputs", out)
            if not isinstance(p, torch.Tensor):
                p = p[0] if isinstance(p, (tuple, list)) else out
            errs.append(float(torch.abs(p.cpu().float() - yb.cpu().float()).mean()))
    return float(np.mean(errs)) if errs else float("nan")


# =============================================================================
# ORCHESTRATOR
# =============================================================================
def _resolve_warm_sd(regime_name: str, regimes: list[RegimeConfig],
                     device, short_warm: str = "padded"):
    """Return the warm-start state dict for a regime, based on its warm_from."""
    by_name = {r.name: r for r in regimes}
    warm_from = regimes[by_name[regime_name].warm_from] if regime_name in by_name else None
    if warm_from is None or regime_name == "pretrained":
        return None
    ckpt = gd.latest_ckpt_in(warm_from.out_dir)
    if ckpt is None:
        return None
    return torch.load(ckpt, map_location=device)


def run_backfill(cfg: BackfillConfig, prices: Optional[pd.DataFrame] = None) -> dict:
    """Run a full backfill job described by `cfg`.

    Phases:
      1. Build windows once (cached per ticker).
      2. For each regime in order:
           - aggregate regimes ("all") train one ckpt over all applicable windows
           - per_ticker regimes train one ckpt per applicable ticker
         Warm-start chains follow regime.warm_from.
      3. Optional comparison: score every per-ticker under global vs per-ticker.
    """
    import torch
    if prices is None:
        prices = pd.read_parquet(PRICES)
    device = _device()
    cb = cfg.callbacks
    log = cb.on_log or (lambda m: print(m, flush=True))

    short_warm = getattr(cfg, "_short_warm", "padded")

    # ---- 1. windows (once) ----
    t0 = time.time()
    wins_by_ticker = build_windows(cfg.data, prices)
    log(f"  built windows for {len(wins_by_ticker)} tickers "
        f"({time.time()-t0:.1f}s, CPU)")
    if cb.on_window_build:
        cb.on_window_build(len(wins_by_ticker), time.time() - t0)

    # ---- 2. regimes ----
    # load heavy base model once; reuse across per-ticker regimes
    base_model, _ = gd.load_granite_model(gd.DEFAULT_MODEL)
    base_model = base_model.to(device)

    # partition tickers into full / short per the plan (driven by data config)
    plan = per_ticker_plan(prices)
    plan = [(tk, n, is_full, sec) for (tk, n, is_full, sec) in plan
            if tk in wins_by_ticker
            and (cfg.data.tickers is None or tk.upper() in {t.upper() for t in cfg.data.tickers})
            and (not cfg.data.exclude_tickers or tk.upper() not in {t.upper() for t in cfg.data.exclude_tickers})]
    full = [(tk, n) for tk, n, is_full, _ in plan if is_full and n >= MIN_NEEDED]
    short = [(tk, n) for tk, n, is_full, _ in plan if (not is_full) and n >= MIN_NEEDED]

    results = {"regimes": {}, "comparison": None}
    for r in cfg.regimes:
        if cb.on_regime:
            cb.on_regime({"regime": r.name, "phase": "start"})
        log(f"=== regime {r.name} (applies_to={r.applies_to}, warm_from={r.warm_from}) ===")
        # Per-regime window config: a regime may override context/horizon/source.
        if r.data is not None:
            rdc = replace(r.data, tickers=cfg.data.tickers,
                          exclude_tickers=cfg.data.exclude_tickers)
            r_wins = build_windows(rdc, prices)
            log(f"  [regime-specific windows] ctx={rdc.context} hor={rdc.horizon} "
                f"n_tk={len(r_wins)}")
        else:
            r_wins = wins_by_ticker
        if r.kind == "aggregate":
            if r.applies_to == "all":
                tickers = full + short
            elif r.applies_to == "full":
                tickers = full
            elif r.applies_to == "short":
                tickers = short
            else:
                tickers = full + short
            if not tickers:
                log(f"  (no tickers for regime {r.name}; skipping)")
                continue
            # subsample per ticker for the aggregate (diversity seed)
            agg_wins = []
            for tk, _ in tickers:
                ws = r_wins[tk]
                if len(ws) > r.agg_windows_per_ticker:
                    idxs = np.linspace(0, len(ws) - 1, r.agg_windows_per_ticker).astype(int)
                    agg_wins += [ws[i] for i in idxs]
                else:
                    agg_wins += ws
            warm_sd = _resolve_warm_sd(r.warm_from, cfg.regimes, device) if r.warm_from else None
            if r.model_config is not None:
                # horizon/context differs from base -> build fresh, no cross-arch warm
                warm_sd = None
            out = train_checkpoint(agg_wins, r.train, device, r.out_dir,
                                   name="granite_ttm_tuned", warm_sd=warm_sd,
                                   model=base_model, cb=cb,
                                   meta={"regime": r.name},
                                   model_config=r.model_config)
            results["regimes"][r.name] = {"out": str(out) if out else None,
                                          "n_windows": len(agg_wins)}
            if cb.on_regime:
                cb.on_regime({"regime": r.name, "phase": "done", "out": str(out)})
        else:
            # per_ticker
            if r.applies_to == "short":
                tk_list = short
                warm_name = short_warm
            else:  # full or all
                tk_list = full if r.applies_to != "all" else (full + short)
                warm_name = r.warm_from or "global"
            warm_regime = next((x for x in cfg.regimes if x.name == warm_name), None)
            warm_sd = (torch.load(gd.latest_ckpt_in(warm_regime.out_dir), map_location=device)
                       if warm_regime and gd.latest_ckpt_in(warm_regime.out_dir) else None)
            total = len(tk_list)
            # For a horizon/context-different regime, build the model ONCE and
            # reuse across tickers (rebuild is expensive); cross-arch warm is off.
            regime_model = None
            if r.model_config is not None:
                warm_sd = None
                regime_model = make_model(device=device, **r.model_config)
            for i, (tk, n) in enumerate(tk_list, 1):
                tw0 = time.time()
                wins = r_wins.get(tk)
                if not wins:
                    continue
                out = train_checkpoint(wins, r.train, device, r.out_dir / tk,
                                       name=f"{tk}_tuned", warm_sd=warm_sd,
                                       model=(regime_model if r.model_config else base_model),
                                       cb=cb,
                                       meta={"regime": r.name, "ticker": tk},
                                       model_config=(r.model_config if regime_model is None else None))
                if cb.on_ticker:
                    cb.on_ticker({"i": i, "n": total, "tk": tk, "wins": len(wins),
                                  "secs": time.time() - tw0})
                if cb.on_ticker and cb.on_ticker({"i": i, "n": total, "tk": tk,
                                                  "wins": len(wins), "secs": time.time() - tw0}) is False:
                    log("  (abort requested via on_ticker)")
                    return results
            results["regimes"][r.name] = {"out_dir": str(r.out_dir), "n_tickers": total}
            if cb.on_regime:
                cb.on_regime({"regime": r.name, "phase": "done", "n": total})

    # ---- 3. comparison ----
    if cfg.compare:
        rows = _write_comparison(cfg, wins_by_ticker, full, short, device, cb)
        results["comparison"] = rows
    return results


def _write_comparison(cfg, wins_by_ticker, full, short, device, cb: Callbacks):
    import torch
    g_regime = next((r for r in cfg.regimes if r.name == "global"), None)
    p_regime = next((r for r in cfg.regimes if r.name == "per_ticker"), None)
    pad_regime = next((r for r in cfg.regimes if r.name == "padded"), None)
    g_ckpt = gd.latest_ckpt_in(g_regime.out_dir) if g_regime else None
    p_ckpt = gd.latest_ckpt_in(p_regime.out_dir) if p_regime else None
    pad_ckpt = gd.latest_ckpt_in(pad_regime.out_dir) if pad_regime else None

    g_model, _ = gd.load_granite_model(gd.DEFAULT_MODEL)
    pt_model, _ = gd.load_granite_model(gd.DEFAULT_MODEL)
    pad_model, _ = gd.load_granite_model(gd.DEFAULT_MODEL)
    if g_ckpt: g_model.load_state_dict(torch.load(g_ckpt, map_location="cpu"))
    if p_ckpt: pt_model.load_state_dict(torch.load(p_ckpt, map_location="cpu"))
    if pad_ckpt: pad_model.load_state_dict(torch.load(pad_ckpt, map_location="cpu"))
    g_model = g_model.to(device); pt_model = pt_model.to(device); pad_model = pad_model.to(device)
    g_model.eval(); pt_model.eval(); pad_model.eval()
    d = date.today().isoformat()

    def err_for(model, tk):
        wins = wins_by_ticker.get(tk, [])
        return score_windows(model, wins, device) if wins else float("nan")

    rows = []
    for tk, n in full:
        pt_ckpt = gd.latest_ckpt_in(p_regime.out_dir / tk) if p_regime else None
        pt_err = float("nan")
        if pt_ckpt:
            pt_model.load_state_dict(torch.load(pt_ckpt, map_location="cpu"))
            pt_model = pt_model.to(device); pt_model.eval()
            pt_err = err_for(pt_model, tk)
        rows.append({"date": d, "ticker": tk, "n": n, "bucket": "full",
                     "global_mae": err_for(g_model, tk), "per_ticker_mae": pt_err,
                     "padded_mae": float("nan")})
    for tk, n in short:
        pt_ckpt = gd.latest_ckpt_in(p_regime.out_dir / tk) if p_regime else None
        pt_err = float("nan")
        if pt_ckpt:
            pt_model.load_state_dict(torch.load(pt_ckpt, map_location="cpu"))
            pt_model = pt_model.to(device); pt_model.eval()
            pt_err = err_for(pt_model, tk)
        rows.append({"date": d, "ticker": tk, "n": n, "bucket": "short",
                     "global_mae": err_for(g_model, tk), "per_ticker_mae": pt_err,
                     "padded_mae": err_for(pad_model, tk)})
    with open(cfg.compare_log, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    full_rows = [r for r in rows if r["bucket"] == "full"]
    per_better = sum(1 for r in full_rows if r["per_ticker_mae"] < r["global_mae"])
    print(f"\n=== backfill comparison ({d}) ===", flush=True)
    print(f"  full tickers: {len(full_rows)} | per-ticker MAE < global MAE on "
          f"{per_better}/{len(full_rows)}", flush=True)
    for r in rows[:8]:
        print(f"    {r['ticker']:5} n={r['n']:4} global={r['global_mae']:.4f} "
              f"per_ticker={r['per_ticker_mae']:.4f} padded={r['padded_mae']:.4f}", flush=True)
    print(f"  full log -> {cfg.compare_log.name}", flush=True)
    if cb.on_compare:
        cb.on_compare(rows)
    return rows


# =============================================================================
# REGIME SWEEP (config-driven parameter exploration)
# =============================================================================
def sweep_regimes(tickers: list[str] | None, base_regime: str,
                  regimes: list[tuple[str, TrainConfig, Optional["DataConfig"],
                                     Optional[dict]]],
                  steps: int | None = None, batch: int = 16,
                  use_adj: bool = False, exclude_tickers: list[str] | None = None,
                  out_root: Path = CKPT_DIR / "sweeps",
                  base_ckpt_dir: Path | None = None) -> dict:
    """Run an arbitrary set of per-ticker training regimes, all warm-started
    from `base_regime` (a named regime whose latest ckpt seeds every sweep
    regime), each saving into its own subdir under `out_root/<name>`.

    This is the config-driven replacement for the copy-pasted pass-3/4 grids:
    each regime is just a (name, TrainConfig[, DataConfig][, model_config]) tuple
    — no loop duplication. DataConfig carries per-regime window overrides
    (context/horizon/price-source); model_config carries per-regime TTM
    architecture overrides (context_length/prediction_length/patch_length/
    use_decoder) for true horizon/context sweeps (the head shape differs, so a
    fresh model is built per regime).

        sweep_regimes(
            tickers=["AEP","NVR"],
            base_regime="per_ticker",
            regimes=[
                ("baseline", TrainConfig(steps=6000), None, None),
                ("hor32",    TrainConfig(steps=6000), DataConfig(horizon=32),
                             {"context_length":512,"prediction_length":32,"patch_length":16,"use_decoder":True}),
                ("lr3e-4",   TrainConfig(steps=6000, lr=3e-4), None, None),
            ],
            use_adj=True,
        )

    Returns the run_backfill results dict.
    """
    cfg = default_backfill_config(use_adj=use_adj, exclude_tickers=list(exclude_tickers or []),
                                  tickers=list(tickers) if tickers else None)
    if base_ckpt_dir is not None:
        for r in cfg.regimes:
            if r.name == base_regime:
                r.out_dir = Path(base_ckpt_dir)
    sweep = []
    for name, tc, dc, mc in regimes:
        if steps is not None:
            tc = replace(tc, steps=steps)
        sweep.append(RegimeConfig(
            name=name, applies_to="full",
            out_dir=out_root / name,
            warm_from=base_regime,
            train=tc, kind="per_ticker",
            data=dc, model_config=mc))   # per-regime window + architecture overrides
    cfg.regimes = [r for r in cfg.regimes if r.name == base_regime] + sweep
    cfg.compare = False
    return run_backfill(cfg)


def compare_adj_unadj(tickers: list[str] | None = None,
                      steps: int = 150, batch: int = 16,
                      ckpt_dir: Path = CKPT_DIR,
                      out_path: Path = HERE / "adj_unadj_compare.jsonl") -> list[dict]:
    """Train (undertrained, `steps`) BOTH an adjusted and an unadjusted per-
    ticker regime for the same tickers, on the SAME windows (sampled identically
    via DataConfig), then report MAPE/MAE for each on identical eval windows.

    The only variable is the price source -> a clean controlled comparison of
    whether adjusted closes help. Evaluation uses build_windows with the same
    max_windows_per_ticker for both, so the test set is identical."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    device = _device()
    prices = pd.read_parquet(PRICES)

    def eval_mape(model, wins):
        ctx = np.stack([w[0] for w in wins])[:, :, None].astype(np.float32)
        tgt = np.stack([w[1] for w in wins])[:, :, None].astype(np.float32)
        ds = TensorDataset(torch.tensor(ctx), torch.tensor(tgt))
        dl = DataLoader(ds, batch_size=batch, shuffle=False)
        model.eval()
        P, A = [], []
        with torch.no_grad():
            for xb, yb in dl:
                xb, yb = xb.to(device), yb.to(device)
                out = model(past_values=xb)
                p = getattr(out, "prediction_outputs", out)
                if not isinstance(p, torch.Tensor):
                    p = p[0] if isinstance(p, (tuple, list)) else out
                P.append(p.cpu().float().numpy()); A.append(yb.cpu().float().numpy())
        P = np.concatenate(P, 0).squeeze(); A = np.concatenate(A, 0).squeeze()
        return float((np.abs(P - A) / np.abs(A).clip(min=1e-6)).mean() * 100)

    rows = []
    tset = {t.upper() for t in tickers} if tickers else None
    for use_adj in (False, True):
        dc = DataConfig(use_adj=use_adj, max_windows_per_ticker=200,
                        tickers=list(tset) if tset else None, out_dir=ckpt_dir)
        wins_by_ticker = build_windows(dc, prices)
        for tk, wins in wins_by_ticker.items():
            if not wins:
                continue
            m, _ = gd.load_granite_model(gd.DEFAULT_MODEL)
            m = m.to(device)
            ctx = np.stack([w[0] for w in wins])[:, :, None]
            tgt = np.stack([w[1] for w in wins])[:, :, None]
            ds = TensorDataset(torch.tensor(ctx), torch.tensor(tgt))
            dl = DataLoader(ds, batch_size=batch, shuffle=True, pin_memory=True)
            m.train()
            opt = torch.optim.AdamW(m.parameters(), lr=gd.LR)
            s = 0
            while s < steps:
                for xb, yb in dl:
                    xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                    o = m(past_values=xb, future_values=yb); loss = o.loss
                    if not torch.isfinite(loss):
                        s = steps; break
                    opt.zero_grad(); loss.backward(); opt.step(); s += 1
                    if s >= steps: break
                if s >= steps: break
            mae_pct = eval_mape(m, wins)
            rows.append({"ticker": tk, "use_adj": use_adj,
                         "n_windows": len(wins), "mape_pct": round(mae_pct, 2)})
            print(f"  {tk} adj={use_adj} mape={mae_pct:.2f}% (n={len(wins)})", flush=True)
    with open(out_path, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return rows


# =============================================================================
# CLI
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    rr = sub.add_parser("run")
    rr.add_argument("--tickers")
    rr.add_argument("--steps", type=int, default=150)
    rr.add_argument("--batch", type=int, default=None)
    rr.add_argument("--use-adj", action="store_true", help="train on adj_close")
    rr.add_argument("--compare", action="store_true")
    sub.add_parser("coverage")
    cmp = sub.add_parser("cmp-adj-unadj")
    cmp.add_argument("--tickers")
    cmp.add_argument("--steps", type=int, default=150)
    cmp.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    if args.cmd == "coverage":
        from granite_backfill import coverage_report
        coverage_report()
    elif args.cmd == "cmp-adj-unadj":
        t = args.tickers.split(",") if args.tickers else None
        compare_adj_unadj(tickers=t, steps=args.steps, batch=args.batch)
    elif args.cmd == "run":
        cfg = default_backfill_config(
            steps=args.steps, batch=args.batch, use_adj=args.use_adj,
            tickers=args.tickers.split(",") if args.tickers else None)
        cfg.compare = args.compare
        run_backfill(cfg)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
