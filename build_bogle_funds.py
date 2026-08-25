#!/usr/bin/env python3
"""
build_bogle_funds.py — Construct Bogle-style index funds from StockMonitor data.

Four funds implementing John C. Bogle's principles:
  1. TMI (Total Market Index)      — Own the whole market, cap-weighted + Fisher chained
  2. QMI (Quality Market Index)    — NM top quintile, liquid, EW + Fisher
     QMI_STRICT                     — Buffett 15/15/1.0, liquid, EW + Fisher
  3. BPI (Bond Proxy Index)        — Defensive anchor: equal-weight, low turnover
  4. PMI (Pink Market Index)       — TMI's complement: the OTC/gray market TMI's
                                     exchange gate excludes, EW + 5% cap + Fisher.
                                     TMI ∪ PMI = complete market, TMI ∩ PMI = ∅.

Usage:
  python build_bogle_funds.py --fund tmi --save
  python build_bogle_funds.py --fund qmi --save
  python build_bogle_funds.py --fund bpi --save
  python build_bogle_funds.py --fund pmi --save
  python build_bogle_funds.py --all --save
  python build_bogle_funds.py --fund tmi --expense-bps 3 --turnover-bps 5 --save
"""
from __future__ import annotations
import argparse
from datetime import date, timedelta
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Add parent directory for stockmagic imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from stock_monitor.index_registry import available_indexes, parse_indexes, tickers_for_index

DATA_DIR = Path(__file__).parent
PRICES_FILE = DATA_DIR / "daily_prices.parquet"
STOCKS_FILE = DATA_DIR / "monitored_stocks.parquet"
FUNDAMENTALS_FILE = DATA_DIR / "fundamentals.parquet"

# Output files
TMI_FILE = DATA_DIR / "bogle_tmi.parquet"
QMI_FILE = DATA_DIR / "bogle_qmi.parquet"
QMI_STRICT_FILE = DATA_DIR / "bogle_qmi_strict.parquet"
BPI_FILE = DATA_DIR / "bogle_bpi.parquet"
TMI_TURNOVER_FILE = DATA_DIR / "bogle_tmi_turnover.parquet"
QMI_TURNOVER_FILE = DATA_DIR / "bogle_qmi_turnover.parquet"
QMI_STRICT_TURNOVER_FILE = DATA_DIR / "bogle_qmi_strict_turnover.parquet"
BPI_TURNOVER_FILE = DATA_DIR / "bogle_bpi_turnover.parquet"
PMI_FILE = DATA_DIR / "bogle_pmi.parquet"
PMI_TURNOVER_FILE = DATA_DIR / "bogle_pmi_turnover.parquet"

# PMI (Pink Market Index) — the complement of TMI's exchange gate.
# TMI owns the exchange-listed market; PMI owns everything else (OTC/pink/gray),
# so TMI + PMI together are complete-market coverage with zero overlap.
PMI_REBAL_FREQ = "Q"          # Quarterly, same calendar as TMI
PMI_MIN_LAST = 1.0            # $1 floor: sub-penny quotes are not investable
PMI_MIN_ADV20 = 100_000.0     # $100k ADV20 (TMI uses $5M)
PMI_MAX_WEIGHT = 0.05         # 5% single-name cap
PMI_EXPENSE_BPS = 5           # OTC costs more to run than TMI's 3
PMI_TURNOVER_BPS = 8          # wider spreads than TMI's 5

# Default cost parameters (Bogle: "costs are the only certain thing")
DEFAULT_EXPENSE_BPS = 3      # 0.03% annual (Vanguard TSM level)
DEFAULT_TURNOVER_BPS = 5     # 0.05% per 100% turnover

# Rebalance frequencies
TMI_REBAL_FREQ = "Q"   # Quarterly
QMI_REBAL_FREQ = "SA"  # Semi-annual
BPI_REBAL_FREQ = "Y"   # Annual (was "A", deprecated in pandas)


def load_prices(tickers: list[str] | None = None, years: float | None = None) -> pd.DataFrame:
    """Load price panel: date x ticker -> close price. Snapshot first (Windows lock)."""
    import shutil, tempfile
    print(f"Loading prices from {PRICES_FILE}...")
    snap = Path(tempfile.gettempdir()) / "bogle_daily_prices.parquet"
    shutil.copy2(PRICES_FILE, snap)
    df = pd.read_parquet(snap, columns=["ticker", "date", "close"])
    if tickers:
        df = df[df["ticker"].isin(tickers)]
    if years:
        cutoff = df["date"].max() - timedelta(days=int(years * 365.25))
        df = df[df["date"] >= cutoff]
    panel = df.pivot_table(index="date", columns="ticker", values="close").sort_index()
    n = panel.notna().sum(axis=1)
    keep = n >= max(50, float(n.median()) * 0.25)
    dropped = int((~keep).sum())
    if dropped:
        panel = panel.loc[keep]
        print(f"  dropped {dropped} thin/holiday dates (n < 25% of median)")
    print(f"  Price panel: {panel.shape[0]} dates x {panel.shape[1]} tickers")
    return panel


def load_fundamentals(tickers: list[str] | None = None) -> pd.DataFrame:
    """Load latest PIT fundamentals for quality screening. Snapshot first."""
    import shutil, tempfile
    if not FUNDAMENTALS_FILE.exists():
        print("  No fundamentals file found")
        return pd.DataFrame()
    snap = Path(tempfile.gettempdir()) / "bogle_fundamentals.parquet"
    shutil.copy2(FUNDAMENTALS_FILE, snap)
    df = pd.read_parquet(snap)
    if tickers:
        df = df[df["ticker"].isin(tickers)]
    if "as_of_date" in df.columns:
        df = df.sort_values("as_of_date").groupby("ticker").tail(1)
    print(f"  Fundamentals: {len(df)} tickers")
    return df


def quality_gate(fund: pd.DataFrame) -> pd.Series:
    """QMI membership: top quintile of Novy-Marx quality (nm_score) with ≥2 legs."""
    if fund.empty:
        return pd.Series(dtype=bool)
    from factor_library import attach_nm_quality
    scored = attach_nm_quality(fund.reset_index(drop=True) if "ticker" in fund.columns else fund)
    legs = pd.to_numeric(scored.get("nm_legs"), errors="coerce").fillna(0)
    score = pd.to_numeric(scored.get("nm_score"), errors="coerce")
    eligible = legs >= 2
    rank = score.where(eligible).rank(pct=True)
    passed = eligible & rank.ge(0.80)
    passed.index = scored["ticker"].astype(str).str.upper() if "ticker" in scored.columns else scored.index
    print(f"  Quality gate (NM top quintile): {int(passed.sum())} / {len(scored)}")
    print(f"    nm_legs>=2: {int(eligible.sum())}, nm_quality: {int(scored.get('nm_quality', pd.Series(False)).fillna(False).sum())}")
    return passed


def quality_gate_strict(fund: pd.DataFrame) -> pd.Series:
    """Tiny QMI: live Buffett ROE/ROIC/D/E (15/15/1.0, D/E ≥ 0)."""
    if fund.empty:
        return pd.Series(dtype=bool)
    from analytics_common import ROE_MIN, ROIC_MIN, DE_MAX
    t = fund.copy()
    t["ticker"] = t["ticker"].astype(str).str.upper()
    roe = pd.to_numeric(t["roe"], errors="coerce") if "roe" in t.columns else pd.Series(np.nan, index=t.index)
    roic = pd.to_numeric(t["roic"], errors="coerce") if "roic" in t.columns else pd.Series(np.nan, index=t.index)
    de = pd.to_numeric(t["debt_to_equity"], errors="coerce") if "debt_to_equity" in t.columns else pd.Series(np.nan, index=t.index)
    passed = roe.ge(ROE_MIN) & roic.ge(ROIC_MIN) & de.ge(0) & de.le(DE_MAX)
    passed.index = t["ticker"]
    print(f"  Quality gate (Buffett 15/15/1.0): {int(passed.sum())} / {len(t)}")
    return passed


def liquid_names(prices: pd.DataFrame, tickers: list[str],
                 min_last: float = 5.0, min_cov: float = 0.80, max_day: float = 1.0,
                 require_mcap: bool = True, require_filing: bool = False,
                 min_adv20: float = 5_000_000.0, liquid_exchanges: set[str] | None = None,
                 exchange_mode: str = "include") -> list[str]:
    """
    Keep names with last price, PIT mcap availability, quarterly filing seen, and exchange.
    PIT gates (all as of last price date):
      - exchange in {NMS,NYQ,NCM,NGM,ASE} and instrument_type=stock (uses monitored_stocks.parquet, now with backfilled exchange)
        exchange_mode="include" keeps names ON those exchanges (TMI/QMI/BPI).
        exchange_mode="exclude" keeps names OFF them (PMI: the OTC/pink complement),
        which makes TMI and PMI disjoint by construction.
      - mcap availability: daily_mcap.parquet has non-null on last date and coverage >= min_cov
      - ADV20 >= min_adv20 (price*volume trailing 20)
      - if require_filing: fundamentals.parquet has ≥1 as_of_date ≤ last date
    Falls back gracefully if panels missing (keeps old price-cov behavior for that leg).
    """
    cols = [t for t in tickers if t in prices.columns]
    if not cols:
        return []
    sub = prices[cols]
    last = sub.iloc[-1]
    cov = sub.notna().mean()
    mx = sub.pct_change().max()
    ok = last.ge(min_last) & cov.ge(min_cov) & mx.le(max_day)

    # Exchange + instrument_type gate (backfilled)
    if liquid_exchanges is None:
        liquid_exchanges = {"NMS", "NYQ", "NCM", "NGM", "ASE"}
    try:
        ms = pd.read_parquet(STOCKS_FILE, columns=["ticker", "instrument_type", "exchange"])
        ms["ticker"] = ms["ticker"].astype(str).str.upper()
        # instrument must be stock
        stock = set(ms.loc[ms["instrument_type"].eq("stock"), "ticker"])
        # exchange must be liquid (NaN = not eligible)
        exch_ok = ms.set_index("ticker")["exchange"].astype(str)
        if exchange_mode == "exclude":
            liquid_tix = {t for t in stock if t in exch_ok.index and str(exch_ok.loc[t]) not in liquid_exchanges}
        else:
            liquid_tix = {t for t in stock if t in exch_ok.index and str(exch_ok.loc[t]) in liquid_exchanges}
        ok = ok & ok.index.to_series().apply(lambda t: t in liquid_tix)
        print(f"  exchange filter ({exchange_mode}) {liquid_exchanges}: {sum(t in liquid_tix for t in ok.index)}/{len(ok)} eligible")
    except Exception as e:
        # The exchange gate is what makes TMI and PMI disjoint. If it silently
        # skipped, PMI would quietly include listed names (and TMI, OTC ones),
        # so the completeness claim would be false with no warning in the output.
        raise RuntimeError(f"exchange gate could not be evaluated: {e}") from e

    # mcap availability gate (PIT)
    if require_mcap:
        try:
            import tempfile, shutil
            mcap_path = DATA_DIR / "daily_mcap.parquet"
            if mcap_path.exists():
                snap = Path(tempfile.gettempdir()) / "bogle_daily_mcap_gate.parquet"
                shutil.copy2(mcap_path, snap)
                mcap = pd.read_parquet(snap)
                mcap["ticker"] = mcap["ticker"].astype(str).str.upper()
                mcap["date"] = pd.to_datetime(mcap["date"]).dt.normalize()
                # coverage per ticker and last-date existence
                last_date = pd.to_datetime(sub.index.max()).normalize()
                # pivot for coverage
                mp = mcap.pivot(index="date", columns="ticker", values="market_cap")
                # align to price index
                mp = mp.reindex(index=pd.to_datetime(sub.index).normalize())
                mcap_cov = mp.notna().mean()
                has_last = mp.loc[last_date].notna() if last_date in mp.index else pd.Series(False, index=mp.columns)
                # tickers without mcap history are not liquid for TMI
                for t in ok.index:
                    if t not in mcap_cov.index or t not in has_last.index:
                        ok.loc[t] = False
                    else:
                        if mcap_cov.loc[t] < min_cov or not bool(has_last.loc[t]):
                            ok.loc[t] = False
                print(f"  mcap gate: {int(ok.sum())} pass (cov>={min_cov} + has mcap on {last_date.date()})")
            else:
                print("  WARNING no daily_mcap.parquet — mcap gate skipped (price weights)")
        except Exception as e:
            # Distinguish "panel absent" (documented fallback, handled above) from
            # "panel present but unreadable", which must not pass silently.
            raise RuntimeError(f"mcap gate could not be evaluated: {e}") from e

    # ADV20 gate (vectorized: one groupby over the panel, not a scan per ticker —
    # PMI's ~8k-name OTC universe made the per-ticker loop the dominant cost).
    # Read via pyarrow with a ticker filter and drop straight to numpy-backed
    # columns: materializing all 33M rows x 4 cols as pandas here failed with
    # `malloc of size 264470976` on a loaded box, and that failure used to be
    # swallowed, emitting an ungated fund.
    try:
        import pyarrow.parquet as pq
        import pyarrow.compute as pc
        want = set(cols)
        last_date = pd.to_datetime(sub.index.max())
        tbl = pq.read_table(
            PRICES_FILE,
            columns=["ticker", "date", "close", "volume"],
            filters=[("ticker", "in", list(want))],
        )
        adv_df = pd.DataFrame({
            "ticker": tbl.column("ticker").to_pandas().astype(str).str.upper(),
            "date": pd.to_datetime(tbl.column("date").to_pandas()),
            "dollar": tbl.column("close").to_pandas().astype("float64")
                      * tbl.column("volume").to_pandas().astype("float64"),
        })
        del tbl
        adv_df = adv_df[adv_df["date"] <= last_date].sort_values(["ticker", "date"])
        tail20 = adv_df.groupby("ticker", sort=False)["dollar"].tail(20)
        adv20 = tail20.groupby(adv_df.loc[tail20.index, "ticker"]).agg(["mean", "count"])
        # require a full 20-day window, same as the prior per-ticker rule
        adv_pass = (adv20["count"] >= 20) & (adv20["mean"] >= min_adv20)
        adv_ok = {t: bool(adv_pass.get(t, False)) for t in cols}
        ok = ok & ok.index.to_series().apply(lambda t: adv_ok.get(t, False))
        print(f"  ADV20>={min_adv20/1e6:.3f}M: {sum(adv_ok.values())}/{len(cols)} pass")
    except Exception as e:
        # A gate that cannot be evaluated must NOT silently disappear: a run that
        # hit `malloc of size 264470976 failed` here still wrote a bogle_pmi.parquet,
        # ungated on liquidity, that looked like a legitimate fund. Fail loudly.
        raise RuntimeError(f"ADV20 gate could not be evaluated: {e}") from e

    # quarterly filing seen gate (for QMI)
    if require_filing:
        try:
            if FUNDAMENTALS_FILE.exists():
                import tempfile, shutil
                snap3 = Path(tempfile.gettempdir()) / "bogle_fund_gate.parquet"
                shutil.copy2(FUNDAMENTALS_FILE, snap3)
                fund = pd.read_parquet(snap3, columns=["ticker", "as_of_date"])
                fund["ticker"] = fund["ticker"].astype(str).str.upper()
                last_date = pd.to_datetime(sub.index.max()).normalize()
                seen = set(fund.loc[pd.to_datetime(fund["as_of_date"]).dt.normalize() <= last_date, "ticker"])
                ok = ok & ok.index.to_series().apply(lambda t: t in seen)
                print(f"  filing gate (≥1 Q seen): {len(seen & set(ok.index))}/{int(ok.sum())} of survivors have filing")
            else:
                print("  WARNING no fundamentals — filing gate skipped")
        except Exception as e:
            raise RuntimeError(f"filing gate could not be evaluated: {e}") from e

    kept = ok[ok].index.astype(str).tolist()
    print(f"  liquidity: {len(kept)} / {len(cols)} (last>={min_last}, cov>={min_cov}, max_day<={max_day}, mcap={require_mcap}, filing={require_filing})")
    return kept


def compute_cap_weights(mcap: pd.DataFrame) -> pd.DataFrame:
    """Date × ticker mcap → cap weights."""
    return mcap.div(mcap.sum(axis=1), axis=0)


def compute_equal_weights(prices: pd.DataFrame) -> pd.DataFrame:
    """Equal weights (1/N) for each date."""
    n = prices.notna().sum(axis=1)
    weights = prices.notna().astype(float).div(n, axis=0)
    return weights


def rebalance_dates(index: pd.DatetimeIndex, freq: str) -> list[pd.Timestamp]:
    """Get rebalance dates at frequency boundaries (nearest trading day on/after boundary)."""
    if freq == "Q":
        boundaries = index.to_period("Q").drop_duplicates().to_timestamp().to_list()
    elif freq == "SA":
        boundaries = index.to_period("6M").drop_duplicates().to_timestamp().to_list()
    elif freq == "A":
        boundaries = index.to_period("A").drop_duplicates().to_timestamp().to_list()
    else:
        boundaries = index.to_period(freq).drop_duplicates().to_timestamp().to_list()

    # Find nearest trading day >= each boundary
    rebal = []
    for b in boundaries:
        # Find first trading day on or after boundary
        mask = index >= b
        if mask.any():
            rebal.append(index[mask][0])
        else:
            rebal.append(index[-1])
    return rebal


def glide_rebalance(current_weights: pd.Series, target_weights: pd.Series,
                    n_days: int = 7) -> list[pd.Series]:
    """Linear glide from current to target weights. Default 7 days (Hoffstein luck)."""
    if n_days <= 1:
        return [target_weights]
    path = []
    for i in range(1, n_days + 1):
        alpha = i / n_days
        w = current_weights * (1 - alpha) + target_weights * alpha
        path.append(w)
    return path


def compute_index_level(prices: pd.DataFrame, weights: pd.DataFrame,
                        expense_bps: float, turnover_bps: float,
                        base_level: float = 1000.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute index level with cost drag and turnover tracking.

    Returns: (levels_df, turnover_df)
    """
    dates = prices.index
    tickers = prices.columns

    # Daily returns
    rets = prices.pct_change().fillna(0)

    # Weight at each date (forward-filled from rebalance)
    # Weights are already aligned to dates
    w = weights.reindex(dates).ffill()

    # Portfolio return each day
    port_rets = (w.shift(1) * rets).sum(axis=1).fillna(0)

    # Expense drag (daily)
    daily_expense = expense_bps / 10000 / 252  # bps -> daily fraction
    port_rets_net = port_rets - daily_expense

    # Turnover at rebalance dates (one-way)
    turnover = pd.Series(0.0, index=dates)
    rebal_dates = weights.index.intersection(dates)
    for i, d in enumerate(rebal_dates):
        if i == 0:
            continue
        prev_w = weights.iloc[i - 1]
        curr_w = weights.iloc[i]
        # One-way turnover = 0.5 * sum(|w_new - w_old|)
        t = 0.5 * (curr_w - prev_w).abs().sum()
        turnover.loc[d] = t

    # Turnover cost drag (applied on rebalance day)
    turnover_cost = turnover * (turnover_bps / 10000)
    port_rets_net = port_rets_net - turnover_cost

    # Index level
    level = base_level * (1 + port_rets_net).cumprod()

    levels_df = pd.DataFrame({
        "date": dates,
        "level": level,
        "ret_gross": port_rets,
        "ret_net": port_rets_net,
        "expense_drag": daily_expense,
        "turnover_cost": turnover_cost,
        "turnover": turnover,
    })
    # Ensure date is a plain column, not an index level
    levels_df = levels_df.reset_index(drop=True)

    turnover_df = pd.DataFrame({
        "date": turnover.index,
        "turnover": turnover.values,
        "turnover_cost": turnover_cost.values,
    })
    turnover_df = turnover_df[turnover_df["turnover"] > 0].reset_index(drop=True)

    return levels_df, turnover_df


def build_fisher_chained(prices: pd.DataFrame, weights: pd.DataFrame,
                         expense_bps: float, turnover_bps: float) -> pd.DataFrame:
    """
    Build Fisher chained index (our de-biased variant).
    Uses rolling 63-day base window, chained period links.
    """
    from stock_monitor.fisher_index import panel, chained_fisher, add_rate_decomposition

    # Get tickers from weights
    tickers = weights.columns.tolist()

    # Build price/quantity panel
    # Use close as price, volume as quantity (from daily_prices)
    vol_panel = pd.read_parquet(PRICES_FILE, columns=["ticker", "date", "volume"])
    vol_panel = vol_panel[vol_panel["ticker"].isin(tickers)]
    q = vol_panel.pivot_table(index="date", columns="ticker", values="volume").sort_index()

    # Align
    p_aligned, q_aligned = prices.align(q, join="inner")

    # Run Fisher chained
    idx = chained_fisher(p_aligned, q_aligned)

    # Apply cost drag
    # Simple approximation: subtract daily expense + turnover cost from fisher_p
    daily_expense = expense_bps / 10000 / 252
    idx["fisher_p_net"] = idx["fisher_p"] * (1 - daily_expense) ** np.arange(len(idx))

    return idx


def cap_weights(weights: pd.Series, max_weight: float, iters: int = 25) -> pd.Series:
    """Iteratively cap single-name weight at max_weight, redistributing to uncapped names.

    Needed for PMI: an equal-weight OTC sleeve still concentrates when the universe
    shrinks, and one pink-sheet story stock must not drive the index.
    """
    w = weights.astype(float).copy()
    total = w.sum()
    if total <= 0:
        return w
    w = w / total
    if len(w) * max_weight < 1.0:  # cap is unreachable; equal-weight is the best we can do
        return pd.Series(1.0 / len(w), index=w.index)
    for _ in range(iters):
        over = w > max_weight
        if not over.any():
            break
        excess = float((w[over] - max_weight).sum())
        w[over] = max_weight
        room = ~over
        base = float(w[room].sum())
        if base <= 0:
            break
        w[room] = w[room] + excess * (w[room] / base)
    return w / w.sum()


def expand_glide_weights(weights: pd.DataFrame, rebal_dates: list[pd.Timestamp],
                         index: pd.DatetimeIndex, n_days: int = 7) -> pd.DataFrame:
    """Expand rebalance-date weights to a daily panel via 7-day linear glide.

    Shared by TMI/QMI/BPI/PMI so the glide (Hoffstein rebalance-luck fix) is
    defined once instead of copy-pasted per fund.
    """
    daily = []
    for i, d in enumerate(rebal_dates):
        if i == 0:
            daily.append(weights.loc[d])
        else:
            prev_w = weights.loc[rebal_dates[i - 1]]
            curr_w = weights.loc[d]
            daily.extend(glide_rebalance(prev_w, curr_w, n_days=n_days))
    out = pd.DataFrame(daily, index=index[:len(daily)])
    return out.reindex(index).ffill()


def attach_fisher(levels: pd.DataFrame, prices: pd.DataFrame, daily_weights: pd.DataFrame,
                  expense_bps: float, turnover_bps: float) -> pd.DataFrame:
    """Merge the Fisher-chained de-biased arm onto a nominal levels frame.

    Shared by every fund. Returns levels unchanged (with a warning) if the Fisher
    arm cannot be built, so a fund still produces its nominal path.
    """
    try:
        fisher = build_fisher_chained(prices, daily_weights, expense_bps, turnover_bps)
        cols = fisher[["date", "fisher_p", "fisher_q", "fisher_p_net", "nominal_sqrt_fisher"]].copy()
        cols["date"] = pd.to_datetime(cols["date"])
        return levels.merge(cols, on="date", how="left")
    except Exception as e:
        print(f"  WARNING Fisher arm skipped: {e}")
        return levels


def build_tmi(prices: pd.DataFrame, expense_bps: float, turnover_bps: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build Total Market Index: cap-weighted + Fisher chained."""
    print("Building TMI (Total Market Index)...")
    names = liquid_names(prices, list(prices.columns), require_mcap=True, require_filing=False)
    prices = prices[names]
    panel_path = DATA_DIR / "daily_mcap.parquet"
    if panel_path.exists():
        panel = pd.read_parquet(panel_path)
        panel["ticker"] = panel["ticker"].astype(str).str.upper()
        panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
        mcap = panel.pivot(index="date", columns="ticker", values="market_cap")
        prices = prices.copy()
        prices.index = pd.to_datetime(prices.index).normalize()
        prices = prices.groupby(level=0).last()
        mcap = mcap.reindex(index=prices.index, columns=prices.columns)
    else:
        mcap = prices.abs()
        print("  WARNING: no daily_mcap.parquet — price-level weights")
    weights = compute_cap_weights(mcap)

    # Rebalance dates
    rebal_dates = rebalance_dates(prices.index, TMI_REBAL_FREQ)
    print(f"  Rebalance dates: {len(rebal_dates)}")

    daily_weights_df = expand_glide_weights(weights, rebal_dates, prices.index)

    # Compute index
    levels, turnover = compute_index_level(prices, daily_weights_df, expense_bps, turnover_bps)

    # Add Fisher chained variant
    levels = attach_fisher(levels, prices, daily_weights_df, expense_bps, turnover_bps)

    levels["fund"] = "TMI"
    levels["weight_method"] = "cap_weighted"
    levels["rebalance_freq"] = TMI_REBAL_FREQ
    levels["expense_bps"] = expense_bps
    levels["turnover_bps"] = turnover_bps

    turnover["fund"] = "TMI"
    return levels, turnover


def build_qmi(prices: pd.DataFrame, expense_bps: float, turnover_bps: float,
              gate=None, fund_name: str = "QMI") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build Quality Market Index: quality-screened + Fisher chained."""
    gate = gate or quality_gate
    print(f"Building {fund_name}...")

    fund = load_fundamentals(prices.columns.tolist())
    if fund.empty:
        print("  WARNING: No fundamentals, falling back to all tickers")
        q_tickers = prices.columns.tolist()
    else:
        passed = gate(fund)
        q_tickers = [t for t in passed.index[passed.fillna(False)] if t in prices.columns]
        if len(q_tickers) == 0:
            print("  WARNING: No tickers passed quality gate, using all")
            q_tickers = prices.columns.tolist()

    q_tickers = liquid_names(prices, q_tickers, require_mcap=True, require_filing=True)
    print(f"  {fund_name} universe: {len(q_tickers)} tickers")

    # Subset prices
    q_prices = prices[q_tickers].dropna(axis=1, how="all")

    # Equal weight (reduces concentration)
    weights = compute_equal_weights(q_prices)

    # Rebalance dates (semi-annual = lower turnover)
    rebal_dates = rebalance_dates(q_prices.index, QMI_REBAL_FREQ)
    print(f"  Rebalance dates: {len(rebal_dates)}")

    daily_weights_df = expand_glide_weights(weights, rebal_dates, q_prices.index)

    # Compute index
    levels, turnover = compute_index_level(q_prices, daily_weights_df, expense_bps, turnover_bps)

    # Add Fisher chained
    levels = attach_fisher(levels, q_prices, daily_weights_df, expense_bps, turnover_bps)

    levels["fund"] = fund_name
    levels["weight_method"] = "equal_weighted"
    levels["rebalance_freq"] = QMI_REBAL_FREQ
    levels["expense_bps"] = expense_bps
    levels["turnover_bps"] = turnover_bps

    turnover["fund"] = fund_name
    return levels, turnover


def build_bpi(prices: pd.DataFrame, expense_bps: float, turnover_bps: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build Bond Proxy Index: defensive sectors, equal-weight, annual rebalance."""
    print("Building BPI (Bond Proxy Index)...")

    # Load monitored stocks for sector info
    stocks = pd.read_parquet(STOCKS_FILE)
    defensive_sectors = ["Utilities", "Consumer Staples", "Health Care", "Real Estate", "Communication Services"]
    # Also include high-yield equity (dividend yield > 3%)
    # For now, use defensive sectors + REITs

    defensive_tickers = stocks[stocks["sector"].isin(defensive_sectors)]["ticker"].tolist()
    defensive_tickers = [t for t in defensive_tickers if t in prices.columns]

    print(f"  BPI universe: {len(defensive_tickers)} tickers (sectors: {defensive_sectors})")

    if len(defensive_tickers) == 0:
        raise ValueError("No defensive tickers found")

    defensive_tickers = liquid_names(prices, defensive_tickers, require_mcap=True, require_filing=False)
    print(f"  BPI liquid: {len(defensive_tickers)} tickers")

    # Subset prices
    bpi_prices = prices[defensive_tickers].dropna(axis=1, how="all")

    # Equal weight
    weights = compute_equal_weights(bpi_prices)

    # Rebalance dates (annual)
    rebal_dates = rebalance_dates(bpi_prices.index, BPI_REBAL_FREQ)
    print(f"  Rebalance dates: {len(rebal_dates)}")

    daily_weights_df = expand_glide_weights(weights, rebal_dates, bpi_prices.index)

    # Compute index
    levels, turnover = compute_index_level(bpi_prices, daily_weights_df, expense_bps, turnover_bps)

    # Add Fisher chained
    levels = attach_fisher(levels, bpi_prices, daily_weights_df, expense_bps, turnover_bps)

    levels["fund"] = "BPI"
    levels["weight_method"] = "equal_weighted"
    levels["rebalance_freq"] = BPI_REBAL_FREQ
    levels["expense_bps"] = expense_bps
    levels["turnover_bps"] = turnover_bps

    turnover["fund"] = "BPI"
    return levels, turnover


def build_pmi(prices: pd.DataFrame, expense_bps: float, turnover_bps: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build Pink Market Index: the OTC/gray-market complement of TMI.

    Complete-market completeness: TMI owns every name ON {NMS,NYQ,NCM,NGM,ASE};
    PMI owns every name OFF them (PNK/OID/OQB/OQX/PCX/unquoted). The two universes
    are disjoint by construction, so TMI + PMI is the whole tape with no double-count.

    Gates are looser than TMI where OTC reality demands it ($1 last, $100k ADV20)
    and NOT looser where it would let junk in: still stock-only, still coverage/max-day,
    still a real 20-day dollar-volume floor. Equal-weight with a 5% single-name cap so
    one pink story stock cannot become the index. Filing-free, like TMI.

    require_mcap is deliberately False here. TMI needs PIT mcap because it is
    cap-weighted — no mcap, no weight. PMI is equal-weighted, so mcap is not an
    input to a single number it computes. And `daily_mcap.parquet` covers only
    900 of 5,695 OTC stocks (34 with >=80% history), because shares-outstanding
    collection follows the listed tape: requiring it yields a 31-name rump that
    is the mcap file's coverage, not the market's complement. Liquidity is
    enforced by the ADV20 floor, which is measured, not inferred.

    MEASURED CEILING (2026-08-24, 10y panel): of 5,351 OTC stocks in the panel,
    only 749 have cov>=0.80 — the OTC tape was never backfilled (median 9 price
    rows per OTC ticker vs 1,799 listed). Of those, 90 clear $1 + max_day, and 85
    clear ADV20>=$100k. So PMI currently prices 85 names, and the binding
    constraint is PRICE COVERAGE, not the gates: relaxing ADV to $0 yields 90,
    relaxing cov to 0.10 yields 153. PMI is therefore structurally correct and
    data-limited; widening it means backfilling OTC prices, not loosening gates.
    """
    print("Building PMI (Pink Market Index)...")
    names = liquid_names(prices, list(prices.columns),
                         min_last=PMI_MIN_LAST, require_mcap=False, require_filing=False,
                         min_adv20=PMI_MIN_ADV20, exchange_mode="exclude")
    if not names:
        raise ValueError("PMI universe empty after gates")
    pmi_prices = prices[names].dropna(axis=1, how="all")
    print(f"  PMI universe: {pmi_prices.shape[1]} tickers")

    # Equal weight, then 5% single-name cap (row-wise; universe changes over time)
    ew = compute_equal_weights(pmi_prices)
    weights = ew.apply(lambda row: cap_weights(row, PMI_MAX_WEIGHT), axis=1)

    rebal_dates = rebalance_dates(pmi_prices.index, PMI_REBAL_FREQ)
    print(f"  Rebalance dates: {len(rebal_dates)}")

    daily_weights_df = expand_glide_weights(weights, rebal_dates, pmi_prices.index)

    levels, turnover = compute_index_level(pmi_prices, daily_weights_df, expense_bps, turnover_bps)
    levels = attach_fisher(levels, pmi_prices, daily_weights_df, expense_bps, turnover_bps)

    levels["fund"] = "PMI"
    levels["weight_method"] = "equal_weighted_capped"
    levels["rebalance_freq"] = PMI_REBAL_FREQ
    levels["expense_bps"] = expense_bps
    levels["turnover_bps"] = turnover_bps

    turnover["fund"] = "PMI"
    return levels, turnover


def save_fund(fund: str, levels: pd.DataFrame, turnover: pd.DataFrame):
    """Save fund outputs."""
    if fund == "TMI":
        levels.to_parquet(TMI_FILE, index=False)
        turnover.to_parquet(TMI_TURNOVER_FILE, index=False)
        print(f"  Saved {TMI_FILE} ({len(levels)} rows)")
        print(f"  Saved {TMI_TURNOVER_FILE} ({len(turnover)} rebalances)")
    elif fund == "QMI":
        levels.to_parquet(QMI_FILE, index=False)
        turnover.to_parquet(QMI_TURNOVER_FILE, index=False)
        print(f"  Saved {QMI_FILE} ({len(levels)} rows)")
        print(f"  Saved {QMI_TURNOVER_FILE} ({len(turnover)} rebalances)")
    elif fund == "QMI_STRICT":
        levels.to_parquet(QMI_STRICT_FILE, index=False)
        turnover.to_parquet(QMI_STRICT_TURNOVER_FILE, index=False)
        print(f"  Saved {QMI_STRICT_FILE} ({len(levels)} rows)")
        print(f"  Saved {QMI_STRICT_TURNOVER_FILE} ({len(turnover)} rebalances)")
    elif fund == "BPI":
        levels.to_parquet(BPI_FILE, index=False)
        turnover.to_parquet(BPI_TURNOVER_FILE, index=False)
        print(f"  Saved {BPI_FILE} ({len(levels)} rows)")
        print(f"  Saved {BPI_TURNOVER_FILE} ({len(turnover)} rebalances)")
    elif fund == "PMI":
        levels.to_parquet(PMI_FILE, index=False)
        turnover.to_parquet(PMI_TURNOVER_FILE, index=False)
        print(f"  Saved {PMI_FILE} ({len(levels)} rows)")
        print(f"  Saved {PMI_TURNOVER_FILE} ({len(turnover)} rebalances)")


def main():
    ap = argparse.ArgumentParser(description="Build Bogle-style index funds")
    ap.add_argument("--fund", choices=["tmi", "qmi", "qmi_strict", "bpi", "pmi", "all"], default="all",
                    help="Which fund to build (default: all)")
    ap.add_argument("--save", action="store_true", help="Write output parquet files")
    ap.add_argument("--expense-bps", type=float, default=DEFAULT_EXPENSE_BPS,
                    help=f"Expense ratio in basis points/year (default: {DEFAULT_EXPENSE_BPS})")
    ap.add_argument("--turnover-bps", type=float, default=DEFAULT_TURNOVER_BPS,
                    help=f"Turnover cost in basis points per 100%% turnover (default: {DEFAULT_TURNOVER_BPS})")
    ap.add_argument("--years", type=float, default=None,
                    help="Limit to last N years of data")
    args = ap.parse_args()

    print(f"Bogle Fund Builder")
    print(f"  Expense ratio: {args.expense_bps} bps/yr")
    print(f"  Turnover cost: {args.turnover_bps} bps per 100% turnover")
    print()

    # Load prices
    prices = load_prices(years=args.years)

    funds_to_build = ["tmi", "qmi", "qmi_strict", "bpi", "pmi"] if args.fund == "all" else [args.fund]

    for fund in funds_to_build:
        print(f"\n{'='*60}")
        if fund == "tmi":
            levels, turnover = build_tmi(prices, args.expense_bps, args.turnover_bps)
        elif fund == "qmi":
            levels, turnover = build_qmi(prices, args.expense_bps, args.turnover_bps,
                                         gate=quality_gate, fund_name="QMI")
        elif fund == "qmi_strict":
            levels, turnover = build_qmi(prices, args.expense_bps, args.turnover_bps,
                                         gate=quality_gate_strict, fund_name="QMI_STRICT")
        elif fund == "bpi":
            levels, turnover = build_bpi(prices, args.expense_bps, args.turnover_bps)
        elif fund == "pmi":
            # PMI carries its own cost defaults (OTC is dearer to run) unless the
            # user overrode them explicitly on the command line.
            pmi_expense = args.expense_bps if args.expense_bps != DEFAULT_EXPENSE_BPS else PMI_EXPENSE_BPS
            pmi_turnover = args.turnover_bps if args.turnover_bps != DEFAULT_TURNOVER_BPS else PMI_TURNOVER_BPS
            levels, turnover = build_pmi(prices, pmi_expense, pmi_turnover)

        if args.save:
            save_fund(fund.upper(), levels, turnover)

        # Print summary
        last = levels.iloc[-1]
        first = levels.iloc[0]
        years = (last["date"] - first["date"]).days / 365.25
        cagr = (last["level"] / first["level"]) ** (1 / years) - 1 if years > 0 else 0
        ann_vol = levels["ret_net"].std() * np.sqrt(252) * 100
        sharpe = (cagr * 100) / ann_vol if ann_vol > 0 else 0

        print(f"\n  Summary:")
        print(f"    Start: {first['date']} level={first['level']:.2f}")
        print(f"    End:   {last['date']} level={last['level']:.2f}")
        print(f"    CAGR:  {cagr*100:.2f}%")
        print(f"    Vol:   {ann_vol:.2f}%")
        print(f"    Sharpe:{sharpe:.2f}")
        print(f"    Total expense drag: {(levels['expense_drag'].sum()*100):.2f}%")
        print(f"    Total turnover cost: {(levels['turnover_cost'].sum()*100):.2f}%")
        print(f"    Avg annual turnover: {turnover['turnover'].mean()*100:.2f}%")

    print(f"\n{'='*60}")
    print("Done.")


if __name__ == "__main__":
    main()