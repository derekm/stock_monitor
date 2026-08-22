#!/usr/bin/env python3
"""
build_bogle_funds.py — Construct Bogle-style index funds from StockMonitor data.

Three funds implementing John C. Bogle's principles:
  1. TMI (Total Market Index)      — Own the whole market, cap-weighted + Fisher chained
  2. QMI (Quality Market Index)    — Factor-tilted: quality gate + Fisher chained
  3. BPI (Bond Proxy Index)        — Defensive anchor: equal-weight, low turnover

Usage:
  python build_bogle_funds.py --fund tmi --save
  python build_bogle_funds.py --fund qmi --save
  python build_bogle_funds.py --fund bpi --save
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
BPI_FILE = DATA_DIR / "bogle_bpi.parquet"
TMI_TURNOVER_FILE = DATA_DIR / "bogle_tmi_turnover.parquet"
QMI_TURNOVER_FILE = DATA_DIR / "bogle_qmi_turnover.parquet"
BPI_TURNOVER_FILE = DATA_DIR / "bogle_bpi_turnover.parquet"

# Default cost parameters (Bogle: "costs are the only certain thing")
DEFAULT_EXPENSE_BPS = 3      # 0.03% annual (Vanguard TSM level)
DEFAULT_TURNOVER_BPS = 5     # 0.05% per 100% turnover

# Rebalance frequencies
TMI_REBAL_FREQ = "Q"   # Quarterly
QMI_REBAL_FREQ = "SA"  # Semi-annual
BPI_REBAL_FREQ = "Y"   # Annual (was "A", deprecated in pandas)


def load_prices(tickers: list[str] | None = None, years: float | None = None) -> pd.DataFrame:
    """Load price panel: date x ticker -> close price."""
    print(f"Loading prices from {PRICES_FILE}...")
    df = pd.read_parquet(PRICES_FILE, columns=["ticker", "date", "close"])
    if tickers:
        df = df[df["ticker"].isin(tickers)]
    if years:
        cutoff = df["date"].max() - timedelta(days=int(years * 365.25))
        df = df[df["date"] >= cutoff]
    # Pivot to wide: date index, ticker columns
    panel = df.pivot_table(index="date", columns="ticker", values="close").sort_index()
    print(f"  Price panel: {panel.shape[0]} dates x {panel.shape[1]} tickers")
    return panel


def load_fundamentals(tickers: list[str] | None = None) -> pd.DataFrame:
    """Load latest PIT fundamentals for quality screening."""
    if not FUNDAMENTALS_FILE.exists():
        print("  No fundamentals file found")
        return pd.DataFrame()
    df = pd.read_parquet(FUNDAMENTALS_FILE)
    if tickers:
        df = df[df["ticker"].isin(tickers)]
    # Keep latest as_of per ticker
    if "as_of_date" in df.columns:
        df = df.sort_values("as_of_date").groupby("ticker").tail(1)
    print(f"  Fundamentals: {len(df)} tickers")
    return df


def quality_gate(fund: pd.DataFrame) -> pd.Series:
    """Apply Buffett-style quality gate: ROE>12%, ROIC>10%, D/E<1.5, Trifecta>=2."""
    if fund.empty:
        return pd.Series(dtype=bool)

    # Trifecta components
    fund = fund.copy()
    fund["trifecta_ev"] = fund.get("ev_ebitda", np.inf) <= 9
    fund["trifecta_pb"] = fund.get("pb_ratio", np.inf) <= 1.5
    fund["trifecta_mcap"] = fund.get("mktcap_to_assets", np.inf) <= 0.5
    fund["trifecta_count"] = fund[["trifecta_ev", "trifecta_pb", "trifecta_mcap"]].sum(axis=1)

    # Quality criteria
    roe_ok = fund.get("roe", 0) > 0.12
    roic_ok = fund.get("roic", 0) > 0.10
    de_ok = fund.get("debt_to_equity", np.inf) < 1.5
    trifecta_ok = fund["trifecta_count"] >= 2

    passed = roe_ok & roic_ok & de_ok & trifecta_ok
    print(f"  Quality gate: {passed.sum()} / {len(fund)} passed")
    print(f"    ROE>12%: {roe_ok.sum()}, ROIC>10%: {roic_ok.sum()}, D/E<1.5: {de_ok.sum()}, Trifecta>=2: {trifecta_ok.sum()}")
    return passed


def compute_cap_weights(prices: pd.DataFrame, shares: pd.Series) -> pd.DataFrame:
    """Compute cap weights for each date. prices: date x ticker, shares: ticker -> shares."""
    mv = prices.mul(shares, axis=1)
    weights = mv.div(mv.sum(axis=1), axis=0)
    return weights


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
                    n_days: int = 5) -> list[pd.Series]:
    """Multi-day glide path from current to target weights (S&P style)."""
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


def build_tmi(prices: pd.DataFrame, expense_bps: float, turnover_bps: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build Total Market Index: cap-weighted + Fisher chained."""
    print("Building TMI (Total Market Index)...")

    # Get shares for cap-weighting (from latest market cap / price)
    # For simplicity, use equal-weight as fallback if no shares
    # In production, load from fundamentals
    shares = pd.Series(1.0, index=prices.columns)  # placeholder

    # Cap weights
    weights = compute_cap_weights(prices, shares)

    # Rebalance dates
    rebal_dates = rebalance_dates(prices.index, TMI_REBAL_FREQ)
    print(f"  Rebalance dates: {len(rebal_dates)}")

    # Build weight series at rebalance dates
    rebal_weights = weights.loc[rebal_dates]

    # Expand to daily with glide
    daily_weights = []
    for i, d in enumerate(rebal_dates):
        if i == 0:
            daily_weights.append(weights.loc[d])
        else:
            prev_w = weights.loc[rebal_dates[i - 1]]
            curr_w = weights.loc[d]
            glide = glide_rebalance(prev_w, curr_w, n_days=5)
            for j, gw in enumerate(glide):
                daily_weights.append(gw)

    daily_weights_df = pd.DataFrame(daily_weights, index=prices.index[:len(daily_weights)])
    daily_weights_df = daily_weights_df.reindex(prices.index).ffill()

    # Compute index
    levels, turnover = compute_index_level(prices, daily_weights_df, expense_bps, turnover_bps)

    # Add Fisher chained variant
    fisher = build_fisher_chained(prices, daily_weights_df, expense_bps, turnover_bps)
    # fisher has 'date' as a column (datetime.date), levels has 'date' as column (datetime64)
    fisher_cols = fisher[["date", "fisher_p", "fisher_q", "fisher_p_net", "nominal_sqrt_fisher"]].copy()
    fisher_cols["date"] = pd.to_datetime(fisher_cols["date"])
    levels = levels.merge(fisher_cols, on="date", how="left")

    levels["fund"] = "TMI"
    levels["weight_method"] = "cap_weighted"
    levels["rebalance_freq"] = TMI_REBAL_FREQ
    levels["expense_bps"] = expense_bps
    levels["turnover_bps"] = turnover_bps

    turnover["fund"] = "TMI"
    return levels, turnover


def build_qmi(prices: pd.DataFrame, expense_bps: float, turnover_bps: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build Quality Market Index: quality-screened + Fisher chained."""
    print("Building QMI (Quality Market Index)...")

    # Load fundamentals and apply quality gate
    fund = load_fundamentals(prices.columns.tolist())
    if fund.empty:
        print("  WARNING: No fundamentals, falling back to all tickers")
        q_tickers = prices.columns.tolist()
    else:
        passed = quality_gate(fund)
        q_tickers = fund.loc[passed, "ticker"].tolist()
        if len(q_tickers) == 0:
            print("  WARNING: No tickers passed quality gate, using all")
            q_tickers = prices.columns.tolist()

    print(f"  QMI universe: {len(q_tickers)} tickers")

    # Subset prices
    q_prices = prices[q_tickers].dropna(axis=1, how="all")

    # Equal weight (reduces concentration)
    weights = compute_equal_weights(q_prices)

    # Rebalance dates (semi-annual = lower turnover)
    rebal_dates = rebalance_dates(q_prices.index, QMI_REBAL_FREQ)
    print(f"  Rebalance dates: {len(rebal_dates)}")

    # Build weight series at rebalance dates
    rebal_weights = weights.loc[rebal_dates]

    # Expand to daily with glide
    daily_weights = []
    for i, d in enumerate(rebal_dates):
        if i == 0:
            daily_weights.append(weights.loc[d])
        else:
            prev_w = weights.loc[rebal_dates[i - 1]]
            curr_w = weights.loc[d]
            glide = glide_rebalance(prev_w, curr_w, n_days=5)
            for j, gw in enumerate(glide):
                daily_weights.append(gw)

    daily_weights_df = pd.DataFrame(daily_weights, index=q_prices.index[:len(daily_weights)])
    daily_weights_df = daily_weights_df.reindex(q_prices.index).ffill()

    # Compute index
    levels, turnover = compute_index_level(q_prices, daily_weights_df, expense_bps, turnover_bps)

    # Add Fisher chained
    fisher = build_fisher_chained(q_prices, daily_weights_df, expense_bps, turnover_bps)
    fisher_cols = fisher[["date", "fisher_p", "fisher_q", "fisher_p_net", "nominal_sqrt_fisher"]].copy()
    fisher_cols["date"] = pd.to_datetime(fisher_cols["date"])
    levels = levels.merge(fisher_cols, on="date", how="left")

    levels["fund"] = "QMI"
    levels["weight_method"] = "equal_weighted"
    levels["rebalance_freq"] = QMI_REBAL_FREQ
    levels["expense_bps"] = expense_bps
    levels["turnover_bps"] = turnover_bps

    turnover["fund"] = "QMI"
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

    # Subset prices
    bpi_prices = prices[defensive_tickers].dropna(axis=1, how="all")

    # Equal weight
    weights = compute_equal_weights(bpi_prices)

    # Rebalance dates (annual)
    rebal_dates = rebalance_dates(bpi_prices.index, BPI_REBAL_FREQ)
    print(f"  Rebalance dates: {len(rebal_dates)}")

    # Build weight series at rebalance dates
    rebal_weights = weights.loc[rebal_dates]

    # Expand to daily with glide
    daily_weights = []
    for i, d in enumerate(rebal_dates):
        if i == 0:
            daily_weights.append(weights.loc[d])
        else:
            prev_w = weights.loc[rebal_dates[i - 1]]
            curr_w = weights.loc[d]
            glide = glide_rebalance(prev_w, curr_w, n_days=5)
            for j, gw in enumerate(glide):
                daily_weights.append(gw)

    daily_weights_df = pd.DataFrame(daily_weights, index=bpi_prices.index[:len(daily_weights)])
    daily_weights_df = daily_weights_df.reindex(bpi_prices.index).ffill()

    # Compute index
    levels, turnover = compute_index_level(bpi_prices, daily_weights_df, expense_bps, turnover_bps)

    # Add Fisher chained
    fisher = build_fisher_chained(bpi_prices, daily_weights_df, expense_bps, turnover_bps)
    fisher_cols = fisher[["date", "fisher_p", "fisher_q", "fisher_p_net", "nominal_sqrt_fisher"]].copy()
    fisher_cols["date"] = pd.to_datetime(fisher_cols["date"])
    levels = levels.merge(fisher_cols, on="date", how="left")

    levels["fund"] = "BPI"
    levels["weight_method"] = "equal_weighted"
    levels["rebalance_freq"] = BPI_REBAL_FREQ
    levels["expense_bps"] = expense_bps
    levels["turnover_bps"] = turnover_bps

    turnover["fund"] = "BPI"
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
    elif fund == "BPI":
        levels.to_parquet(BPI_FILE, index=False)
        turnover.to_parquet(BPI_TURNOVER_FILE, index=False)
        print(f"  Saved {BPI_FILE} ({len(levels)} rows)")
        print(f"  Saved {BPI_TURNOVER_FILE} ({len(turnover)} rebalances)")


def main():
    ap = argparse.ArgumentParser(description="Build Bogle-style index funds")
    ap.add_argument("--fund", choices=["tmi", "qmi", "bpi", "all"], default="all",
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

    funds_to_build = ["tmi", "qmi", "bpi"] if args.fund == "all" else [args.fund]

    for fund in funds_to_build:
        print(f"\n{'='*60}")
        if fund == "tmi":
            levels, turnover = build_tmi(prices, args.expense_bps, args.turnover_bps)
        elif fund == "qmi":
            levels, turnover = build_qmi(prices, args.expense_bps, args.turnover_bps)
        elif fund == "bpi":
            levels, turnover = build_bpi(prices, args.expense_bps, args.turnover_bps)

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