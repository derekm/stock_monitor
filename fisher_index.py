#!/usr/bin/env python3
"""
fisher_index.py — Chained Laspeyres / Paasche / Fisher price & quantity indexes.

Uses daily close as price (p) and volume as quantity (q).

Formulas (link from t-1 → t):
  Laspeyres_P = Σ p_t q_{t-1} / Σ p_{t-1} q_{t-1}
  Paasche_P   = Σ p_t q_t     / Σ p_{t-1} q_t
  Fisher_P    = sqrt(Laspeyres_P * Paasche_P)

  Laspeyres_Q = Σ p_{t-1} q_t / Σ p_{t-1} q_{t-1}
  Paasche_Q   = Σ p_t q_t     / Σ p_t q_{t-1}
  Fisher_Q    = sqrt(Laspeyres_Q * Paasche_Q)

  Nominal value index link ≈ Fisher_P * Fisher_Q
  (also reported: sqrt(Fisher_P * Fisher_Q) as a geometric nominal summary)

Chained levels: 100 * cumulative product of links (base = first date with complete basket).

Usage:
  python fisher_index.py --universe portfolio --save
  python fisher_index.py --universe fertilizer
  python fisher_index.py --sector Materials
  python fisher_index.py --tickers MOS,CF,NTR,SHEL --save
  python fisher_index.py --universe all --freq W --save
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from cli_common import (
    add_index_args, add_ticker_args, add_sector_arg, add_save_arg, add_freq_arg,
    add_window_arg, resolve_tickers_from_args, resolve_index_names_from_args,
    build_parser,
)
from index_registry import parse_indexes, tickers_for_index, available_indexes, index_help_text

DATA_DIR = Path(__file__).parent
PRICES_FILE = DATA_DIR / "daily_prices.parquet"
STOCKS_FILE = DATA_DIR / "monitored_stocks.parquet"
OUT_FILE = DATA_DIR / "fisher_indexes.csv"
OUT_PQ = DATA_DIR / "fisher_indexes.parquet"


def load_pq() -> pd.DataFrame:
    df = pd.read_parquet(PRICES_FILE)
    # `date` is DATE on disk -> read as datetime.date; keep it a date.
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df.get("volume", 0), errors="coerce")
    # leave zeros as NaN so panel can ffill prior session volume as quantity weight
    df.loc[df["volume"] <= 0, "volume"] = pd.NA
    return df


def resolve_tickers(
    universe: Optional[str] = None,
    sector: Optional[str] = None,
    tickers: Optional[str] = None,
) -> list[str]:
    if tickers:
        return [x.strip().upper() for x in tickers.split(",") if x.strip()]
    if sector:
        stocks = pd.read_parquet(STOCKS_FILE) if STOCKS_FILE.exists() else pd.DataFrame()
        if not stocks.empty and "sector" in stocks.columns:
            return stocks.loc[stocks["sector"].str.lower() == sector.lower(), "ticker"].tolist()
    if universe:
        try:
            names = parse_indexes(universe)
        except ValueError as e:
            raise SystemExit(str(e)) from e
        seen, out = set(), []
        for n in names:
            for tk in tickers_for_index(n):
                if tk not in seen:
                    seen.add(tk)
                    out.append(tk)
        return out
    return tickers_for_index("fertilizer") or ["MOS", "CF", "SHEL"]



def panel(prices: pd.DataFrame, tickers: list[str], freq: str = "D") -> tuple[pd.DataFrame, pd.DataFrame]:
    sub = prices[prices["ticker"].isin(tickers)].copy()
    p = sub.pivot_table(index="date", columns="ticker", values="close").sort_index()
    q = sub.pivot_table(index="date", columns="ticker", values="volume").sort_index()
    # align
    p, q = p.align(q, join="inner")
    if freq and freq.upper() != "D":
        # period-end price, sum volume.
        # resample requires a DatetimeIndex, so promote locally, then drop
        # back to datetime.date so the panel index stays a date.
        p = p.set_axis(pd.to_datetime(p.index))
        q = q.set_axis(pd.to_datetime(q.index))
        p = p.resample(freq).last()
        q = q.resample(freq).sum()
        p = p.set_axis([d.date() for d in p.index])
        q = q.set_axis([d.date() for d in q.index])
        p, q = p.align(q, join="inner")
    # drop days with all-null prices
    mask = p.notna().any(axis=1)
    p, q = p.loc[mask], q.loc[mask]
    p = p.ffill()
    q = q.replace(0, np.nan)
    # Prefer prior volume, then trailing median — never drop to 1 unless no history
    q = q.ffill().bfill()
    q_med = q.rolling(21, min_periods=1).median()
    q = q.fillna(q_med).fillna(1.0)
    return p, q


def chained_fisher(p: pd.DataFrame, q: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame of chained index levels and links."""
    dates = p.index
    cols = p.columns
    rows = []
    # levels start at 100 on first complete observation
    Lp = Lq = Pp = Pq = Fp = Fq = 100.0
    nom_prod = 100.0
    nom_geom = 100.0

    for i in range(len(dates)):
        if i == 0:
            rows.append({
                "date": dates[i],
                "laspeyres_p": 100.0, "paasche_p": 100.0, "fisher_p": 100.0,
                "laspeyres_q": 100.0, "paasche_q": 100.0, "fisher_q": 100.0,
                "nominal_fisher_product": 100.0,
                "nominal_sqrt_fisher": 100.0,
                "link_laspeyres_p": 1.0, "link_paasche_p": 1.0, "link_fisher_p": 1.0,
                "link_laspeyres_q": 1.0, "link_paasche_q": 1.0, "link_fisher_q": 1.0,
                "n_items": int((p.iloc[i].notna() & (q.iloc[i] > 0)).sum()),
            })
            continue

        p0, p1 = p.iloc[i - 1], p.iloc[i]
        q0, q1 = q.iloc[i - 1], q.iloc[i]
        # carry prior volume when current session volume is zero/missing
        q1 = q1.where(q1 > 0, q0)
        q0 = q0.where(q0 > 0, q1)
        ok = p0.notna() & p1.notna() & q0.notna() & q1.notna() & (q0 > 0) & (q1 > 0)
        p0, p1, q0, q1 = p0[ok], p1[ok], q0[ok], q1[ok]
        if len(p0) == 0:
            link_lp = link_pp = link_fp = 1.0
            link_lq = link_pq = link_fq = 1.0
        else:
            # avoid div0
            def safe_div(num, den):
                den = den if den != 0 else np.nan
                return float(num / den) if den and np.isfinite(den) and den != 0 else 1.0

            sum_p1q0 = float((p1 * q0).sum())
            sum_p0q0 = float((p0 * q0).sum())
            sum_p1q1 = float((p1 * q1).sum())
            sum_p0q1 = float((p0 * q1).sum())

            link_lp = safe_div(sum_p1q0, sum_p0q0)  # Laspeyres P
            link_pp = safe_div(sum_p1q1, sum_p0q1)  # Paasche P
            link_fp = float(np.sqrt(max(link_lp, 0) * max(link_pp, 0))) if link_lp > 0 and link_pp > 0 else 1.0

            link_lq = safe_div(sum_p0q1, sum_p0q0)  # Laspeyres Q
            link_pq = safe_div(sum_p1q1, sum_p1q0)  # Paasche Q
            link_fq = float(np.sqrt(max(link_lq, 0) * max(link_pq, 0))) if link_lq > 0 and link_pq > 0 else 1.0

        Lp *= link_lp
        Pp *= link_pp
        Fp *= link_fp
        Lq *= link_lq
        Pq *= link_pq
        Fq *= link_fq
        nom_prod *= (link_fp * link_fq)
        nom_geom *= float(np.sqrt(max(link_fp, 0) * max(link_fq, 0))) if link_fp > 0 and link_fq > 0 else 1.0

        rows.append({
            "date": dates[i],
            "laspeyres_p": Lp, "paasche_p": Pp, "fisher_p": Fp,
            "laspeyres_q": Lq, "paasche_q": Pq, "fisher_q": Fq,
            "nominal_fisher_product": nom_prod,
            "nominal_sqrt_fisher": nom_geom,
            "link_laspeyres_p": link_lp, "link_paasche_p": link_pp, "link_fisher_p": link_fp,
            "link_laspeyres_q": link_lq, "link_paasche_q": link_pq, "link_fisher_q": link_fq,
            "n_items": int(ok.sum()) if hasattr(ok, "sum") else len(p0),
        })
    return pd.DataFrame(rows)



def add_rate_decomposition(idx: pd.DataFrame, periods_per_year: float = 252.0) -> pd.DataFrame:
    """Inflation vs growth decomposition from Fisher links and levels.

    Exact period identity on *links* (product form):
        (1 + π_t) * (1 + g_t) = link_fisher_p * link_fisher_q
        ≈ nominal value growth factor for the period

    Where:
      π_t  = link_fisher_p - 1   (period inflation / price growth)
      g_t  = link_fisher_q - 1   (period real / quantity growth)

    Cumulative levels (base 100 at first row):
      Fisher_P, Fisher_Q, nominal_fisher_product = 100 * Π links

    Also reports trailing and annualized rates from levels.
    """
    out = idx.copy()
    out = out.sort_values("date").reset_index(drop=True)
    # period rates from links when present
    if "link_fisher_p" in out.columns:
        out["infl_rate"] = out["link_fisher_p"] - 1.0
        out["growth_rate"] = out["link_fisher_q"] - 1.0
        out["nominal_rate"] = out["link_fisher_p"] * out["link_fisher_q"] - 1.0
        # residual of identity (should be ~0)
        out["identity_gap"] = (1.0 + out["nominal_rate"]) - (1.0 + out["infl_rate"]) * (1.0 + out["growth_rate"])
    else:
        # fallback from level ratios
        fp = out["fisher_p"].astype(float)
        fq = out["fisher_q"].astype(float)
        out["infl_rate"] = fp.pct_change().fillna(0.0)
        out["growth_rate"] = fq.pct_change().fillna(0.0)
        out["nominal_rate"] = (fp * fq).pct_change().fillna(0.0)
        out["identity_gap"] = (1.0 + out["nominal_rate"]) - (1.0 + out["infl_rate"]) * (1.0 + out["growth_rate"])

    # cumulative contribution indexes (base 100) — same as levels for P/Q product path
    out["cum_infl_factor"] = (1.0 + out["infl_rate"].fillna(0.0)).cumprod()
    out["cum_growth_factor"] = (1.0 + out["growth_rate"].fillna(0.0)).cumprod()
    out["cum_nominal_factor"] = (1.0 + out["nominal_rate"].fillna(0.0)).cumprod()

    # trailing 21-period annualized rates from levels
    win = min(21, max(5, len(out) // 10))
    for col, src in [
        ("infl_ann_21", "fisher_p"),
        ("growth_ann_21", "fisher_q"),
        ("nominal_ann_21", "nominal_fisher_product"),
    ]:
        s = out[src].astype(float)
        ratio = s / s.shift(win)
        out[col] = (ratio ** (periods_per_year / win) - 1.0) * 100.0  # percent

    return out


def rebase_to_date(idx: pd.DataFrame, ref_date: str | pd.Timestamp, level_cols: list[str] | None = None) -> pd.DataFrame:
    """Rebase selected level columns so ref_date = 100."""
    out = idx.copy()
    # `date` is already datetime.date (carried through from the panel index),
    # so the parquet `date` column is a DATE type with no casting.
    ref = pd.to_datetime(ref_date)
    # nearest available date on or before ref, else on or after
    dates = out["date"].sort_values()
    prior = dates[dates <= ref]
    if len(prior):
        use = prior.iloc[-1]
    else:
        after = dates[dates >= ref]
        if not len(after):
            return out
        use = after.iloc[0]
    base_row = out.loc[out["date"] == use].iloc[0]
    cols = level_cols or [
        "laspeyres_p", "paasche_p", "fisher_p",
        "laspeyres_q", "paasche_q", "fisher_q",
        "nominal_fisher_product", "nominal_sqrt_fisher",
    ]
    for c in cols:
        if c in out.columns and pd.notna(base_row.get(c)) and float(base_row[c]) != 0:
            out[c] = out[c].astype(float) / float(base_row[c]) * 100.0
    out["ref_date"] = use.date().isoformat()
    return out



def run(tickers: list[str], freq: str = "D", label: str = "",
        years: float | None = None, ref_date: str | None = None) -> pd.DataFrame:
    prices = load_pq()
    if years is not None and years > 0:
        cutoff = prices["date"].max() - pd.Timedelta(days=int(years * 365.25))
        prices = prices[prices["date"] >= cutoff]
    p, q = panel(prices, tickers, freq=freq)
    if p.empty or len(p) < 3:
        raise SystemExit(f"Insufficient price/volume panel for {tickers[:5]}… (n_dates={len(p)})")
    idx = chained_fisher(p, q)
    idx["universe"] = label or ",".join(tickers[:5])
    idx["freq"] = freq
    idx["n_tickers"] = len(tickers)
    ppy = 252.0 if str(freq).upper() in ("D", "B", "") else (52.0 if str(freq).upper().startswith("W") else 12.0)
    idx = add_rate_decomposition(idx, periods_per_year=ppy)
    if ref_date:
        idx = rebase_to_date(idx, ref_date)
    return idx


def main():
    ap = argparse.ArgumentParser(description="Chained Fisher / Laspeyres / Paasche indexes")
    add_index_args(ap, default="portfolio")
    add_ticker_args(ap)
    add_sector_arg(ap)
    add_freq_arg(ap)
    add_save_arg(ap)
    ap.add_argument("--ref-date", default=None, help="Rebase levels so this date = 100 (YYYY-MM-DD)")
    ap.add_argument("--years", type=float, default=None, help="Only use last N years of prices")
    ap.add_argument("--backfill-all", action="store_true",
                    help="Rebuild portfolio,fertilizer,defensive,growth_tech,Materials and save")
    args = ap.parse_args()

    def _save_merge(frames: list[pd.DataFrame]):
        out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if out.empty:
            return
        if OUT_FILE.exists() and not args.backfill_all:
            old = pd.read_csv(OUT_FILE)
            old["date"] = old["date"].apply(
                lambda s: datetime.strptime(str(s)[:10], "%Y-%m-%d").date())
            for _, g in out.groupby(["universe", "freq"], dropna=False):
                lab, fr = g["universe"].iloc[0], g["freq"].iloc[0]
                old = old[~((old["universe"] == lab) & (old["freq"] == fr))]
            out = pd.concat([old, out], ignore_index=True)
        elif OUT_FILE.exists() and args.backfill_all:
            # replace all rebuilt universes
            old = pd.read_csv(OUT_FILE)
            old["date"] = old["date"].apply(
                lambda s: datetime.strptime(str(s)[:10], "%Y-%m-%d").date())
            labs = set(out["universe"].unique())
            old = old[~old["universe"].isin(labs)]
            out = pd.concat([old, out], ignore_index=True)
        out.to_csv(OUT_FILE, index=False)
        try:
            out.to_parquet(OUT_PQ, index=False)
        except Exception:
            pass
        # also write decomposition-focused CSV
        rate_cols = [c for c in out.columns if c in (
            "date", "universe", "freq", "fisher_p", "fisher_q", "nominal_fisher_product",
            "nominal_sqrt_fisher", "infl_rate", "growth_rate", "nominal_rate", "identity_gap",
            "infl_ann_21", "growth_ann_21", "nominal_ann_21", "ref_date", "n_items", "n_tickers",
        )]
        out[rate_cols].to_csv(DATA_DIR / "fisher_rate_decomposition.csv", index=False)
        print(f"Wrote {OUT_FILE} and fisher_rate_decomposition.csv ({len(out)} rows)")

    jobs = []
    if args.backfill_all:
        jobs = [
            ("portfolio", None),
            ("fertilizer", None),
            ("defensive", None),
            ("growth_tech", None),
            ("Materials", "Materials"),
        ]
    else:
        jobs = [(None, args.sector)]

    frames = []
    for idx_name, sector in jobs:
        if sector:
            tickers = resolve_tickers(sector=sector)
            label = sector
        elif idx_name:
            # force single index name
            class _A: pass
            a = _A(); a.index = idx_name; a.universe = idx_name; a.tickers = None; a.sector = None
            tickers = resolve_tickers_from_args(a, default_index=idx_name)
            label = idx_name
        else:
            tickers = resolve_tickers_from_args(args, default_index="portfolio")
            label = args.sector or (",".join(resolve_index_names_from_args(args, default_index="portfolio")) or "custom")
        if not tickers:
            print(f"Skip {label}: no tickers")
            continue
        print(f"Building chained indexes for {len(tickers)} names ({label}), freq={args.freq}")
        idx = run(tickers, freq=args.freq, label=label, years=args.years, ref_date=args.ref_date)
        print(idx[["date", "fisher_p", "fisher_q", "nominal_sqrt_fisher", "infl_rate", "growth_rate"]].tail(5).to_string(index=False))
        last = idx.iloc[-1]
        print(
            f"  Last: Fp={last.fisher_p:.2f} Fq={last.fisher_q:.2f}  "
            f"π≈{last.infl_rate*100:.3f}%  g≈{last.growth_rate*100:.3f}%  "
            f"(1+π)(1+g)-1≈{last.nominal_rate*100:.3f}%  gap={last.identity_gap:.2e}"
        )
        frames.append(idx)

    if args.save or args.backfill_all:
        _save_merge(frames)


if __name__ == "__main__":
    main()
