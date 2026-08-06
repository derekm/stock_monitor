#!/usr/bin/env python3
"""
peer_analytics.py — Cross-stock peer comparison analytics.

Automates the RF-style analysis across all stocks:
- Peer group mapping (sector + analytics groups)
- Fundamental trend detection (ROE, ROIC, earnings stability, P/B)
- Risk-adjusted peer rankings (Sharpe, beta, vol vs peers)
- Recovery detection (improving fundamentals after deterioration)
- Signal generation for pipeline integration

Outputs:
- peer_analytics_signals.csv — per-stock signals for pipeline
- peer_group_summary.csv — group-level statistics
- peer_fundamental_trends.csv — trend slopes and significance
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import polars as pl

DATA_DIR = Path(__file__).parent
PRICES = DATA_DIR / "daily_prices.parquet"
STOCKS = DATA_DIR / "monitored_stocks.parquet"
FUND = DATA_DIR / "fundamentals.parquet"
OUT_SIGNALS = DATA_DIR / "peer_analytics_signals.csv"
OUT_GROUP = DATA_DIR / "peer_group_summary.csv"
OUT_TRENDS = DATA_DIR / "peer_fundamental_trends.csv"
OUT_RECOVERY = DATA_DIR / "peer_recovery_signals.csv"

# Fundamental metrics to track trends for
FUND_METRICS = [
    "roe", "roic", "debt_to_equity", "ev_ebitda", "pb_ratio",
    "earnings_stability", "interest_coverage", "mktcap_to_assets"
]

# Risk metrics to compute
RISK_WINDOWS = [21, 63, 126, 252]


def load_data() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Load prices, stocks metadata, and fundamentals."""
    prices = pl.read_parquet(PRICES, columns=["date", "ticker", "adj_close", "close"])
    stocks = pl.read_parquet(STOCKS)
    fund = pl.read_parquet(FUND)
    return prices, stocks, fund


def get_peer_groups(stocks: pl.DataFrame) -> dict[str, list[str]]:
    """
    Build peer groups from sector and analytics group columns.
    Returns dict mapping group_name -> list of tickers.
    """
    groups = {}

    # Sector groups
    if "sector" in stocks.columns:
        for sector in stocks["sector"].unique():
            if sector and str(sector).strip():
                tickers = stocks.filter(pl.col("sector") == sector)["ticker"].to_list()
                if len(tickers) >= 3:  # minimum group size
                    groups[f"sector_{sector}"] = tickers

    # Analytics group (growth_tech sleeves)
    if "growth_sleeve" in stocks.columns:
        for sleeve in stocks["growth_sleeve"].unique():
            if sleeve and str(sleeve).strip():
                tickers = stocks.filter(pl.col("growth_sleeve") == sleeve)["ticker"].to_list()
                if len(tickers) >= 3:
                    groups[f"sleeve_{sleeve}"] = tickers

    # Value sleeves
    if "value_sleeve" in stocks.columns:
        for sleeve in stocks["value_sleeve"].unique():
            if sleeve and str(sleeve).strip():
                tickers = stocks.filter(pl.col("value_sleeve") == sleeve)["ticker"].to_list()
                if len(tickers) >= 3:
                    groups[f"value_{sleeve}"] = tickers

    # Defensive index
    if "defensive_value_index" in stocks.columns:
        tickers = stocks.filter(pl.col("defensive_value_index") == True)["ticker"].to_list()
        if len(tickers) >= 3:
            groups["defensive_index"] = tickers

    # Growth tech index
    if "growth_tech_index" in stocks.columns:
        tickers = stocks.filter(pl.col("growth_tech_index") == True)["ticker"].to_list()
        if len(tickers) >= 3:
            groups["growth_tech_index"] = tickers

    return groups


def compute_rolling_returns(prices: pl.DataFrame, windows: list[int]) -> dict[str, pl.DataFrame]:
    """Compute rolling returns for each window."""
    # Pivot to wide format for vectorized computation
    wide = prices.select(["date", "ticker", "adj_close"]).pivot(
        index="date", columns="ticker", values="adj_close"
    ).sort("date")

    tickers = [c for c in wide.columns if c != "date"]
    date_col = wide["date"].to_numpy()
    results = {}

    for window in windows:
        # Compute log returns over window
        rets_dict = {"date": date_col[window:]}
        for t in tickers:
            vals = wide[t].to_numpy()
            if len(vals) > window:
                rets = np.log(vals[window:] / vals[:-window])
                rets_dict[t] = rets
        results[f"ret_{window}d"] = pl.DataFrame(rets_dict)

    return results


def compute_rolling_vol(prices: pl.DataFrame, windows: list[int]) -> dict[str, pl.DataFrame]:
    """Compute rolling volatility for each window."""
    wide = prices.select(["date", "ticker", "adj_close"]).pivot(
        index="date", columns="ticker", values="adj_close"
    ).sort("date")

    tickers = [c for c in wide.columns if c != "date"]
    date_col = wide["date"].to_numpy()
    results = {}

    for window in windows:
        vol_dict = {"date": date_col[window:]}
        for t in tickers:
            vals = wide[t].to_numpy()
            if len(vals) > window:
                logrets = np.log(vals[1:] / vals[:-1])
                # Rolling std * sqrt(252/window) for annualized
                roll_vol = pd.Series(logrets).rolling(window).std().values * np.sqrt(252)
                vol_dict[t] = roll_vol[window-1:]
        results[f"vol_{window}d"] = pl.DataFrame(vol_dict)

    return results


def compute_beta_to_group(prices: pl.DataFrame, group_tickers: list[str], window: int = 126) -> dict[str, float]:
    """Compute beta of each stock to its peer group equal-weight return."""
    wide = prices.select(["date", "ticker", "adj_close"]).pivot(
        index="date", columns="ticker", values="adj_close"
    ).sort("date")

    # Filter to tickers that exist in data
    available = [t for t in group_tickers if t in wide.columns]
    if len(available) < 3:
        return {}

    # Group return (equal weight)
    group_rets = np.log(wide[available].to_numpy()[1:] / wide[available].to_numpy()[:-1]).mean(axis=1)

    results = {}
    for t in available:
        if t in wide.columns:
            stock_rets = np.log(wide[t].to_numpy()[1:] / wide[t].to_numpy()[:-1])
            if len(stock_rets) >= window and len(group_rets) >= window:
                # Rolling beta
                roll_betas = []
                for i in range(window, len(stock_rets)):
                    s = stock_rets[i-window:i]
                    g = group_rets[i-window:i]
                    if np.std(g) > 0:
                        beta = np.cov(s, g)[0,1] / np.var(g)
                    else:
                        beta = 1.0
                    roll_betas.append(beta)
                if roll_betas:
                    results[t] = float(np.nanmean(roll_betas[-63:]))  # recent 63d avg beta
    return results


def analyze_fundamental_trends(fund: pl.DataFrame, min_obs: int = 4) -> pl.DataFrame:
    """
    Compute trend slopes for fundamental metrics per ticker.
    Returns DataFrame with ticker, metric, slope, pct_change, n_obs.
    """
    fund_sorted = fund.sort(["ticker", "as_of_date"])
    results = []

    for ticker in fund_sorted["ticker"].unique():
        t_data = fund_sorted.filter(pl.col("ticker") == ticker)
        if len(t_data) < min_obs:
            continue

        dates = t_data["as_of_date"].to_numpy()
        # Convert dates to numeric (days since first)
        date_nums = np.arange(len(dates), dtype=float)

        for metric in FUND_METRICS:
            if metric not in t_data.columns:
                continue
            vals = t_data[metric].to_numpy()
            # Remove NaN
            mask = ~pd.isna(vals)
            if mask.sum() < min_obs:
                continue
            x = date_nums[mask]
            y = vals[mask]

            # Linear regression
            if len(x) > 1 and np.std(x) > 0:
                slope = np.cov(x, y)[0,1] / np.var(x)
                intercept = np.mean(y) - slope * np.mean(x)
                # Total change over period
                total_change = slope * (x[-1] - x[0])
                pct_change = total_change / y[0] if y[0] != 0 else np.nan

                # Recent vs early (last 2 vs first 2)
                recent_avg = np.mean(y[-2:]) if len(y) >= 2 else y[-1]
                early_avg = np.mean(y[:2]) if len(y) >= 2 else y[0]
                recent_vs_early = (recent_avg - early_avg) / early_avg if early_avg != 0 else np.nan

                results.append({
                    "ticker": ticker,
                    "metric": metric,
                    "slope_per_period": float(slope),
                    "total_change": float(total_change),
                    "pct_change": float(pct_change) if not np.isnan(pct_change) else None,
                    "recent_vs_early_pct": float(recent_vs_early) if not np.isnan(recent_vs_early) else None,
                    "n_obs": int(mask.sum()),
                    "latest_value": float(y[-1]),
                    "earliest_value": float(y[0]),
                })

    return pl.DataFrame(results) if results else pl.DataFrame()


def detect_recovery(trends: pl.DataFrame) -> pl.DataFrame:
    """
    Detect recovery patterns: metrics that deteriorated then improved.
    Recovery = metric was declining (negative slope overall) but recent trend positive.
    """
    if len(trends) == 0:
        return pl.DataFrame()

    results = []
    for ticker in trends["ticker"].unique():
        t_data = trends.filter(pl.col("ticker") == ticker)
        for metric in FUND_METRICS:
            m_data = t_data.filter(pl.col("metric") == metric)
            if len(m_data) == 0:
                continue
            row = m_data.row(0, named=True)

            overall_slope = row["slope_per_period"]
            recent_vs_early = row["recent_vs_early_pct"]
            latest = row["latest_value"]
            earliest = row["earliest_value"]

            # Recovery signal: overall negative slope but recent improvement
            is_recovery = (overall_slope < 0) and (recent_vs_early is not None) and (recent_vs_early > 0.05)

            # Deterioration signal: overall positive slope but recent decline
            is_deteriorating = (overall_slope > 0) and (recent_vs_early is not None) and (recent_vs_early < -0.05)

            # Strong trend (consistent direction)
            strong_trend = abs(overall_slope) > 0.01 and (recent_vs_early is not None) and \
                          np.sign(overall_slope) == np.sign(recent_vs_early)

            results.append({
                "ticker": ticker,
                "metric": metric,
                "overall_slope": overall_slope,
                "recent_vs_early_pct": recent_vs_early,
                "latest_value": latest,
                "is_recovery": is_recovery,
                "is_deteriorating": is_deteriorating,
                "strong_trend": strong_trend,
                "trend_direction": "improving" if overall_slope > 0 else "declining" if overall_slope < 0 else "flat",
            })

    return pl.DataFrame(results)


def compute_peer_rankings(
    prices: pl.DataFrame,
    fund: pl.DataFrame,
    stocks: pl.DataFrame,
    peer_groups: dict[str, list[str]],
    windows: list[int] = [63, 126, 252]
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Compute peer-relative rankings for each stock within each group.
    Returns (signals_df, group_summary_df).
    """
    # Get latest fundamentals
    latest_fund = fund.sort("as_of_date").group_by("ticker").tail(1)

    # Compute returns and vol for ranking windows
    returns_dict = compute_rolling_returns(prices, windows)
    vol_dict = compute_rolling_vol(prices, windows)

    all_signals = []
    all_group_stats = []

    for group_name, tickers in peer_groups.items():
        available = [t for t in tickers if t in latest_fund["ticker"].to_list()]
        if len(available) < 3:
            continue

        # Group-level stats
        group_fund = latest_fund.filter(pl.col("ticker").is_in(available))

        for metric in FUND_METRICS:
            if metric in group_fund.columns:
                vals = group_fund[metric].to_numpy()
                vals = vals[~pd.isna(vals)]
                if len(vals) > 0:
                    all_group_stats.append({
                        "group": group_name,
                        "metric": metric,
                        "n": len(vals),
                        "mean": float(np.mean(vals)),
                        "median": float(np.median(vals)),
                        "std": float(np.std(vals)),
                        "p25": float(np.percentile(vals, 25)),
                        "p75": float(np.percentile(vals, 75)),
                    })

        # Per-stock rankings within group
        for window in windows:
            ret_key = f"ret_{window}d"
            vol_key = f"vol_{window}d"

            if ret_key in returns_dict and vol_key in vol_dict:
                rets_df = returns_dict[ret_key]
                vols_df = vol_dict[vol_key]

                for ticker in available:
                    if ticker not in rets_df.columns or ticker not in vols_df.columns:
                        continue

                    # Get recent return and vol
                    ticker_ret = rets_df[ticker].drop_nulls().to_numpy()
                    ticker_vol = vols_df[ticker].drop_nulls().to_numpy()

                    if len(ticker_ret) == 0 or len(ticker_vol) == 0:
                        continue

                    latest_ret = ticker_ret[-1] if len(ticker_ret) > 0 else np.nan
                    latest_vol = ticker_vol[-1] if len(ticker_vol) > 0 else np.nan

                    # Group peers' latest values
                    peer_rets = []
                    peer_vols = []
                    for p in available:
                        if p != ticker and p in rets_df.columns and p in vols_df.columns:
                            pr = rets_df[p].drop_nulls().to_numpy()
                            pv = vols_df[p].drop_nulls().to_numpy()
                            if len(pr) > 0 and len(pv) > 0:
                                peer_rets.append(pr[-1])
                                peer_vols.append(pv[-1])

                    if len(peer_rets) < 2:
                        continue

                    peer_rets = np.array(peer_rets)
                    peer_vols = np.array(peer_vols)

                    # Percentile ranks
                    ret_rank = (peer_rets < latest_ret).mean() if not np.isnan(latest_ret) else 0.5
                    vol_rank = (peer_vols > latest_vol).mean() if not np.isnan(latest_vol) else 0.5  # lower vol = better rank

                    # Risk-adjusted rank (Sharpe-like)
                    if latest_vol > 0:
                        sharpe = latest_ret / latest_vol
                        peer_sharpes = peer_rets / peer_vols
                        sharpe_rank = (peer_sharpes < sharpe).mean()
                    else:
                        sharpe_rank = 0.5

                    all_signals.append({
                        "ticker": ticker,
                        "group": group_name,
                        "window": window,
                        "ret": float(latest_ret) if not np.isnan(latest_ret) else None,
                        "vol": float(latest_vol) if not np.isnan(latest_vol) else None,
                        "ret_rank": float(ret_rank),
                        "vol_rank": float(vol_rank),
                        "sharpe_rank": float(sharpe_rank),
                        "n_peers": len(peer_rets),
                    })

    signals_df = pl.DataFrame(all_signals) if all_signals else pl.DataFrame()
    group_df = pl.DataFrame(all_group_stats) if all_group_stats else pl.DataFrame()
    return signals_df, group_df


def compute_beta_signals(
    prices: pl.DataFrame,
    peer_groups: dict[str, list[str]]
) -> pl.DataFrame:
    """Compute beta to peer group and flag high-beta names."""
    all_betas = []

    for group_name, tickers in peer_groups.items():
        betas = compute_beta_to_group(prices, tickers, window=126)
        if not betas:
            continue

        beta_vals = np.array(list(betas.values()))
        beta_median = np.median(beta_vals)
        beta_p75 = np.percentile(beta_vals, 75)

        for ticker, beta in betas.items():
            all_betas.append({
                "ticker": ticker,
                "group": group_name,
                "beta_to_group": beta,
                "beta_vs_median": beta - beta_median,
                "beta_pctile": (beta_vals < beta).mean(),
                "high_beta_flag": beta > beta_p75,
                "low_beta_flag": beta < np.percentile(beta_vals, 25),
            })

    return pl.DataFrame(all_betas) if all_betas else pl.DataFrame()


def generate_signals(
    trends: pl.DataFrame,
    recovery: pl.DataFrame,
    peer_signals: pl.DataFrame,
    beta_signals: pl.DataFrame,
    latest_fund: pl.DataFrame,
    stocks: pl.DataFrame
) -> pl.DataFrame:
    """Aggregate all signals into per-ticker actionable signals."""
    # Start with latest fundamentals + stock metadata
    stock_cols = ["ticker"]
    for c in ["sector", "growth_sleeve", "value_sleeve", "defensive_value_index", "growth_tech_index"]:
        if c in stocks.columns:
            stock_cols.append(c)
    
    signals = stocks.select(stock_cols).unique()
    # Join with latest fund data
    signals = signals.join(latest_fund.select(["ticker"] + [c for c in FUND_METRICS if c in latest_fund.columns]), on="ticker", how="left")

    # Add fundamental trends
    if len(trends) > 0:
        # Pivot trends to wide format (only columns that exist in trends)
        trend_value_cols = [c for c in ["slope_per_period", "recent_vs_early_pct", "latest_value", "pct_change", "total_change", "n_obs"] if c in trends.columns]
        if trend_value_cols:
            trend_wide = trends.pivot(
                values=trend_value_cols,
                index="ticker", columns="metric"
            )
            signals = signals.join(trend_wide, on="ticker", how="left")

    # Add recovery signals
    if len(recovery) > 0:
        # Pivot recovery to wide format (only columns that exist)
        rec_value_cols = [c for c in ["is_recovery", "is_deteriorating", "strong_trend", "trend_direction", "overall_slope", "recent_vs_early_pct", "latest_value"] if c in recovery.columns]
        if rec_value_cols:
            rec_wide = recovery.pivot(
                values=rec_value_cols,
                index="ticker", columns="metric"
            )
            signals = signals.join(rec_wide, on="ticker", how="left")

    # Add best peer rankings (use 126d window as primary)
    if len(peer_signals) > 0:
        primary = peer_signals.filter(pl.col("window") == 126)
        if len(primary) > 0:
            # Best group for each ticker (highest sharpe_rank)
            best = primary.sort("sharpe_rank", descending=True).group_by("ticker").head(1)
            best = best.select(["ticker", "group", "ret_rank", "vol_rank", "sharpe_rank", "n_peers"])
            best = best.rename({
                "group": "best_peer_group",
                "ret_rank": "best_ret_rank",
                "vol_rank": "best_vol_rank",
                "sharpe_rank": "best_sharpe_rank",
                "n_peers": "best_n_peers",
            })
            signals = signals.join(best, on="ticker", how="left")

    # Add beta signals
    if len(beta_signals) > 0:
        # Average beta across groups
        beta_avg = beta_signals.group_by("ticker").agg(
            pl.col("beta_to_group").mean().alias("avg_beta_to_peers"),
            pl.col("high_beta_flag").any().alias("high_beta_any_group"),
            pl.col("low_beta_flag").any().alias("low_beta_any_group"),
            pl.col("beta_vs_median").mean().alias("avg_beta_vs_median"),
        )
        signals = signals.join(beta_avg, on="ticker", how="left")

    # Composite signal scoring
    signals = signals.with_columns([
        # Recovery score: count of recovering metrics
        pl.sum_horizontal([
            pl.col(f"is_recovery_{m}").fill_null(False).cast(pl.Int32) for m in FUND_METRICS
            if f"is_recovery_{m}" in signals.columns
        ]).alias("recovery_count"),

        # Deterioration score
        pl.sum_horizontal([
            pl.col(f"is_deteriorating_{m}").fill_null(False).cast(pl.Int32) for m in FUND_METRICS
            if f"is_deteriorating_{m}" in signals.columns
        ]).alias("deterioration_count"),

        # Strong trend count
        pl.sum_horizontal([
            pl.col(f"strong_trend_{m}").fill_null(False).cast(pl.Int32) for m in FUND_METRICS
            if f"strong_trend_{m}" in signals.columns
        ]).alias("strong_trend_count"),
    ])

    # Signal classification
    signals = signals.with_columns([
        pl.when(pl.col("recovery_count") >= 2)
        .then(pl.lit("RECOVERING"))
        .when(pl.col("deterioration_count") >= 2)
        .then(pl.lit("DETERIORATING"))
        .when(pl.col("strong_trend_count") >= 3)
        .then(pl.lit("STRONG_TREND"))
        .otherwise(pl.lit("NEUTRAL"))
        .alias("fundamental_signal"),

        # Peer-relative signal
        pl.when(pl.col("best_sharpe_rank") >= 0.75)
        .then(pl.lit("PEER_LEADER"))
        .when(pl.col("best_sharpe_rank") <= 0.25)
        .then(pl.lit("PEER_LAGGARD"))
        .otherwise(pl.lit("PEER_AVERAGE"))
        .alias("peer_signal"),

        # Beta signal
        pl.when(pl.col("high_beta_any_group"))
        .then(pl.lit("HIGH_BETA"))
        .when(pl.col("low_beta_any_group"))
        .then(pl.lit("LOW_BETA"))
        .otherwise(pl.lit("NEUTRAL_BETA"))
        .alias("beta_signal"),
    ])

    return signals


def run(save: bool = True) -> dict[str, pl.DataFrame]:
    """Main entry point."""
    print("Loading data...")
    prices, stocks, fund = load_data()
    print(f"  Prices: {len(prices)} rows, {prices['ticker'].n_unique()} tickers")
    print(f"  Stocks: {len(stocks)} rows")
    print(f"  Fundamentals: {len(fund)} rows, {fund['ticker'].n_unique()} tickers")

    print("Building peer groups...")
    peer_groups = get_peer_groups(stocks)
    print(f"  Found {len(peer_groups)} peer groups:")
    for k, v in peer_groups.items():
        print(f"    {k}: {len(v)} tickers")

    print("Analyzing fundamental trends...")
    trends = analyze_fundamental_trends(fund)
    print(f"  Computed trends for {trends['ticker'].n_unique()} tickers x {len(FUND_METRICS)} metrics")

    print("Detecting recovery patterns...")
    recovery = detect_recovery(trends)
    n_recovery = recovery.filter(pl.col("is_recovery")).height if len(recovery) > 0 else 0
    n_deteriorating = recovery.filter(pl.col("is_deteriorating")).height if len(recovery) > 0 else 0
    print(f"  Recovery signals: {n_recovery}, Deteriorating: {n_deteriorating}")

    print("Computing peer rankings...")
    peer_signals, group_summary = compute_peer_rankings(prices, fund, stocks, peer_groups)
    print(f"  Peer signals: {len(peer_signals)} rows, Group stats: {len(group_summary)} rows")

    print("Computing beta signals...")
    beta_signals = compute_beta_signals(prices, peer_groups)
    print(f"  Beta signals: {len(beta_signals)} rows")

    print("Generating composite signals...")
    latest_fund = fund.sort("as_of_date").group_by("ticker").tail(1)
    signals = generate_signals(trends, recovery, peer_signals, beta_signals, latest_fund, stocks)
    print(f"  Final signals: {len(signals)} tickers")

    # Signal distribution
    if "fundamental_signal" in signals.columns:
        print("\n  Fundamental signals:")
        for sig, count in signals.group_by("fundamental_signal").agg(pl.count()).iter_rows():
            print(f"    {sig}: {count}")

    if "peer_signal" in signals.columns:
        print("\n  Peer signals:")
        for sig, count in signals.group_by("peer_signal").agg(pl.count()).iter_rows():
            print(f"    {sig}: {count}")

    if save:
        print(f"\nSaving outputs...")
        signals.write_csv(OUT_SIGNALS)
        group_summary.write_csv(OUT_GROUP)
        trends.write_csv(OUT_TRENDS)
        recovery.write_csv(OUT_RECOVERY)
        print(f"  {OUT_SIGNALS}")
        print(f"  {OUT_GROUP}")
        print(f"  {OUT_TRENDS}")
        print(f"  {OUT_RECOVERY}")

    return {
        "signals": signals,
        "group_summary": group_summary,
        "trends": trends,
        "recovery": recovery,
        "peer_signals": peer_signals,
        "beta_signals": beta_signals,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true", default=True)
    args = ap.parse_args()
    run(save=args.save)


if __name__ == "__main__":
    main()