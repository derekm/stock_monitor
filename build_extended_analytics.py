#!/usr/bin/env python3
"""
Extended look-through fundamentals and price analytics using ownership network.

Computes for each filer × quarter:
1. Extended look-through fundamentals (all available metrics)
2. Portfolio price analytics (returns, risk, factor exposures)
3. Attribution analysis (holding-level contribution)
4. Concentration and diversification metrics
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from datetime import date, timedelta

# Load data
print("Loading data...")
quarterly_edges = pd.read_parquet('quarterly_network_edges.parquet')
quarterly_edges['as_of_date'] = pd.to_datetime(quarterly_edges['as_of_date']).dt.date

fund = pd.read_parquet('fundamentals.parquet')
fund['as_of_date'] = pd.to_datetime(fund['as_of_date']).dt.date

prices = pd.read_parquet('daily_prices.parquet')
prices['date'] = pd.to_datetime(prices['date']).dt.date

# All fundamental metrics available
FUNDAMENTAL_METRICS = [
    'ev_ebitda', 'roic', 'fcf_margin', 'debt_to_equity', 'interest_coverage',
    'roe', 'reinvestment_rate', 'pb_ratio', 'mktcap_to_assets',
    'earnings_stability', 'free_cash_flow', 'capital_expenditure_ttm',
    'total_debt', 'shareholders_equity', 'total_assets', 'revenue_quarterly',
    'market_cap', 'market_cap_b'
]

# Filter to available
available_metrics = [m for m in FUNDAMENTAL_METRICS if m in fund.columns]
print(f"Available fundamental metrics: {available_metrics}")

quarters = sorted(quarterly_edges['as_of_date'].unique())
filers = quarterly_edges['filer_ticker'].unique()
held_tickers = quarterly_edges['held_ticker'].unique()

print(f"Quarters: {len(quarters)} ({quarters[0]} to {quarters[-1]})")
print(f"Filers: {len(filers)}")
print(f"Held tickers: {len(held_tickers)}")

# ============================================================
# 1. EXTENDED LOOK-THROUGH FUNDAMENTALS
# ============================================================
print("\n=== EXTENDED LOOK-THROUGH FUNDAMENTALS ===")

# Prepare fundamental data - latest available as of each quarter
fund_by_ticker = fund.set_index(['ticker', 'as_of_date'])[available_metrics].sort_index()

lt_results = []

for q in tqdm(quarters, desc="Look-through fundamentals"):
    q_edges = quarterly_edges[quarterly_edges['as_of_date'] == q]
    if len(q_edges) == 0:
        continue
    
    # Get fundamental data as of this quarter (latest prior or equal)
    fund_q = fund[fund['as_of_date'] <= q].drop_duplicates(subset=['ticker', 'as_of_date'], keep='last')
    fund_q = fund_q.set_index('ticker')[available_metrics]
    
    for filer in q_edges['filer_ticker'].unique():
        f_edges = q_edges[q_edges['filer_ticker'] == filer]
        total_val = f_edges['market_value'].sum()
        
        if total_val == 0:
            continue
        
        # Weight by market_value
        weights = f_edges.set_index('held_ticker')['market_value'] / total_val
        
        # Align with fundamentals
        common = weights.index.intersection(fund_q.index)
        if len(common) == 0:
            continue
        
        w = weights[common]
        w = w / w.sum()  # renormalize to available
        
        lt_vals = {'filer_ticker': filer, 'as_of_date': q, 'holdings_value': total_val}
        
        for metric in available_metrics:
            vals = fund_q.loc[common, metric]
            # Remove NaN
            valid = vals.dropna()
            if len(valid) > 0:
                w_valid = w[valid.index]
                w_valid = w_valid / w_valid.sum()
                lt_vals[f'lt_{metric}'] = float(np.sum(w_valid * valid))
                lt_vals[f'lt_{metric}_coverage'] = float(w_valid.sum())  # weight covered
            else:
                lt_vals[f'lt_{metric}'] = np.nan
                lt_vals[f'lt_{metric}_coverage'] = 0.0
        
        lt_vals['n_holdings_with_fund'] = len(valid)
        lt_results.append(lt_vals)

lt_df = pd.DataFrame(lt_results)
lt_df.to_parquet('quarterly_lookthrough_fundamentals_extended.parquet', index=False)
print(f"Saved {len(lt_df)} rows to quarterly_lookthrough_fundamentals_extended.parquet")

# ============================================================
# 2. PORTFOLIO PRICE ANALYTICS
# ============================================================
print("\n=== PORTFOLIO PRICE ANALYTICS ===")

# For each filer × quarter, compute portfolio returns/risk over next quarter
# We'll use daily prices for the quarter following as_of_date

# Get price data for all held tickers
held_price_data = prices[prices['ticker'].isin(held_tickers)].copy()
held_price_data = held_price_data.sort_values(['ticker', 'date'])

# Compute daily returns
held_price_data['daily_return'] = held_price_data.groupby('ticker')['adj_close'].pct_change()

price_results = []

for q in tqdm(quarters, desc="Price analytics"):
    q_edges = quarterly_edges[quarterly_edges['as_of_date'] == q]
    if len(q_edges) == 0:
        continue
    
    # Define quarter window: next 63 trading days (~3 months)
    q_start = q
    q_end = q + timedelta(days=120)  # generous window
    
    # Get prices in this window
    window_prices = held_price_data[
        (held_price_data['date'] >= q_start) & 
        (held_price_data['date'] <= q_end)
    ].copy()
    
    if len(window_prices) == 0:
        continue
    
    for filer in q_edges['filer_ticker'].unique():
        f_edges = q_edges[q_edges['filer_ticker'] == filer]
        total_val = f_edges['market_value'].sum()
        
        if total_val == 0:
            continue
        
        weights = f_edges.set_index('held_ticker')['market_value'] / total_val
        
        # Get returns for holdings
        filer_tickers = f_edges['held_ticker'].unique()
        filer_returns = window_prices[window_prices['ticker'].isin(filer_tickers)][
            ['date', 'ticker', 'daily_return']
        ].dropna()
        
        if len(filer_returns) == 0:
            continue
        
        # Pivot to date × ticker matrix
        ret_matrix = filer_returns.pivot(index='date', columns='ticker', values='daily_return')
        ret_matrix = ret_matrix.fillna(0)
        
        # Align weights
        common_tickers = ret_matrix.columns.intersection(weights.index)
        if len(common_tickers) == 0:
            continue
        
        w = weights[common_tickers]
        w = w / w.sum()
        
        ret_matrix = ret_matrix[common_tickers]
        
        # Portfolio daily returns
        port_returns = (ret_matrix * w).sum(axis=1)
        
        # Analytics
        n_days = len(port_returns)
        if n_days < 10:
            continue
        
        # Cumulative return
        cum_return = (1 + port_returns).prod() - 1
        
        # Volatility (annualized)
        vol_daily = port_returns.std()
        vol_ann = vol_daily * np.sqrt(252)
        
        # Sharpe (assuming 0 risk-free for simplicity)
        sharpe = port_returns.mean() / vol_daily * np.sqrt(252) if vol_daily > 0 else 0
        
        # Max drawdown
        cum = (1 + port_returns).cumprod()
        running_max = cum.expanding().max()
        drawdown = (cum - running_max) / running_max
        max_dd = drawdown.min()
        
        # Skewness, kurtosis
        skew = port_returns.skew()
        kurt = port_returns.kurtosis()
        
        # VaR (5%)
        var_95 = port_returns.quantile(0.05)
        cvar_95 = port_returns[port_returns <= var_95].mean() if (port_returns <= var_95).any() else var_95
        
        # Portfolio beta vs SPY (if available)
        spy_returns = window_prices[window_prices['ticker'] == 'SPY'][['date', 'daily_return']].dropna()
        beta = np.nan
        if len(spy_returns) > 10:
            spy_ret = spy_returns.set_index('date')['daily_return']
            common_dates = port_returns.index.intersection(spy_ret.index)
            if len(common_dates) > 10:
                cov = np.cov(port_returns.loc[common_dates], spy_ret.loc[common_dates])[0, 1]
                var_spy = np.var(spy_ret.loc[common_dates])
                beta = cov / var_spy if var_spy > 0 else np.nan
        
        # Holding-level attribution
        holding_contrib = {}
        for t in common_tickers:
            h_ret = ret_matrix[t].mean() * n_days  # approx period return
            h_contrib = w[t] * h_ret
            holding_contrib[t] = h_contrib
        
        # Top/bottom contributors
        top_contrib = sorted(holding_contrib.items(), key=lambda x: x[1], reverse=True)[:5]
        bot_contrib = sorted(holding_contrib.items(), key=lambda x: x[1])[:5]
        
        # Factor exposures (simple: growth/value, size, momentum via holdings)
        # Growth exposure = weighted avg PB ratio (inverse)
        # Size exposure = weighted avg log market cap
        # Momentum = weighted avg 12m return
        
        price_results.append({
            'filer_ticker': filer,
            'as_of_date': q,
            'holdings_value': total_val,
            'n_holdings': len(common_tickers),
            'period_days': n_days,
            'cum_return': cum_return,
            'vol_annualized': vol_ann,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'skewness': skew,
            'kurtosis': kurt,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'beta_spy': beta,
            'top_contributors': json.dumps(top_contrib),
            'bottom_contributors': json.dumps(bot_contrib)
        })

price_df = pd.DataFrame(price_results)
price_df.to_parquet('quarterly_portfolio_analytics.parquet', index=False)
print(f"Saved {len(price_df)} rows to quarterly_portfolio_analytics.parquet")

# ============================================================
# 3. CONCENTRATION & DIVERSIFICATION METRICS
# ============================================================
print("\n=== CONCENTRATION & DIVERSIFICATION METRICS ===")

conc_results = []

for q in tqdm(quarters, desc="Concentration metrics"):
    q_edges = quarterly_edges[quarterly_edges['as_of_date'] == q]
    if len(q_edges) == 0:
        continue
    
    for filer in q_edges['filer_ticker'].unique():
        f_edges = q_edges[q_edges['filer_ticker'] == filer]
        total_val = f_edges['market_value'].sum()
        
        if total_val == 0:
            continue
        
        weights = f_edges['market_value'] / total_val
        
        # HHI
        hhi = (weights ** 2).sum()
        
        # Effective number of holdings
        n_eff = 1 / hhi if hhi > 0 else 0
        
        # Top-K concentration
        top5 = weights.nlargest(5).sum()
        top10 = weights.nlargest(10).sum()
        top20 = weights.nlargest(20).sum()
        
        # Entropy (Shannon)
        entropy = -(weights * np.log(weights + 1e-10)).sum()
        max_entropy = np.log(len(weights))
        entropy_norm = entropy / max_entropy if max_entropy > 0 else 0
        
        # Gini coefficient
        sorted_w = np.sort(weights)
        n = len(sorted_w)
        gini = (2 * np.arange(1, n+1) @ sorted_w) / (n * sorted_w.sum()) - (n + 1) / n
        
        # Sector concentration (if sector data available)
        # For now, use industry from nodes
        
        conc_results.append({
            'filer_ticker': filer,
            'as_of_date': q,
            'holdings_value': total_val,
            'n_holdings': len(weights),
            'hhi': hhi,
            'n_eff': n_eff,
            'top5_concentration': top5,
            'top10_concentration': top10,
            'top20_concentration': top20,
            'entropy': entropy,
            'entropy_normalized': entropy_norm,
            'gini': gini
        })

conc_df = pd.DataFrame(conc_results)
conc_df.to_parquet('quarterly_concentration_metrics.parquet', index=False)
print(f"Saved {len(conc_df)} rows to quarterly_concentration_metrics.parquet")

# ============================================================
# 4. FACTOR EXPOSURES THROUGH HOLDINGS
# ============================================================
print("\n=== FACTOR EXPOSURES ===")

# Use fundamentals as factor proxies:
# - Value: 1/PB, 1/EV_EBITDA
# - Quality: ROIC, ROE, FCF margin
# - Size: log(market_cap)
# - Momentum: 12m return (from prices)
# - Leverage: debt_to_equity

factor_results = []

for q in tqdm(quarters, desc="Factor exposures"):
    q_edges = quarterly_edges[quarterly_edges['as_of_date'] == q]
    if len(q_edges) == 0:
        continue
    
    # Get fundamental data as of quarter
    fund_q = fund[fund['as_of_date'] <= q].drop_duplicates(subset=['ticker', 'as_of_date'], keep='last')
    fund_q = fund_q.set_index('ticker')
    
    # Get 12m momentum from prices
    q_start = q - timedelta(days=365)
    mom_prices = prices[
        (prices['date'] >= q_start) & 
        (prices['date'] <= q) &
        (prices['ticker'].isin(held_tickers))
    ].copy()
    mom_prices = mom_prices.sort_values(['ticker', 'date'])
    
    # 12m return per ticker
    mom_returns = mom_prices.groupby('ticker')['adj_close'].apply(
        lambda x: x.iloc[-1] / x.iloc[0] - 1 if len(x) > 1 else np.nan
    )
    
    for filer in q_edges['filer_ticker'].unique():
        f_edges = q_edges[q_edges['filer_ticker'] == filer]
        total_val = f_edges['market_value'].sum()
        
        if total_val == 0:
            continue
        
        weights = f_edges.set_index('held_ticker')['market_value'] / total_val
        
        # Align with fundamentals
        common = weights.index.intersection(fund_q.index)
        if len(common) == 0:
            continue
        
        w = weights[common]
        w = w / w.sum()
        
        # Factor exposures (weighted averages)
        exposures = {'filer_ticker': filer, 'as_of_date': q}
        
        # Value factor
        if 'pb_ratio' in fund_q.columns:
            pb = fund_q.loc[common, 'pb_ratio'].dropna()
            if len(pb) > 0:
                wp = w[pb.index]
                wp = wp / wp.sum()
                exposures['factor_value_pb'] = float((wp * (1/pb)).sum())
        
        if 'ev_ebitda' in fund_q.columns:
            ev = fund_q.loc[common, 'ev_ebitda'].dropna()
            if len(ev) > 0:
                wp = w[ev.index]
                wp = wp / wp.sum()
                exposures['factor_value_ev'] = float((wp * (1/ev)).sum())
        
        # Quality factor
        for qual in ['roic', 'roe', 'fcf_margin']:
            if qual in fund_q.columns:
                vals = fund_q.loc[common, qual].dropna()
                if len(vals) > 0:
                    wp = w[vals.index]
                    wp = wp / wp.sum()
                    exposures[f'factor_quality_{qual}'] = float((wp * vals).sum())
        
        # Size factor
        if 'market_cap' in fund_q.columns:
            mc = fund_q.loc[common, 'market_cap'].dropna()
            if len(mc) > 0:
                wp = w[mc.index]
                wp = wp / wp.sum()
                exposures['factor_size_log_mc'] = float((wp * np.log(mc)).sum())
        
        # Momentum factor
        mom_common = common.intersection(mom_returns.index)
        if len(mom_common) > 0:
            mom_vals = mom_returns[mom_common].dropna()
            if len(mom_vals) > 0:
                wp = w[mom_vals.index]
                wp = wp / wp.sum()
                exposures['factor_momentum_12m'] = float((wp * mom_vals).sum())
        
        # Leverage factor
        if 'debt_to_equity' in fund_q.columns:
            dte = fund_q.loc[common, 'debt_to_equity'].dropna()
            if len(dte) > 0:
                wp = w[dte.index]
                wp = wp / wp.sum()
                exposures['factor_leverage'] = float((wp * dte).sum())
        
        # Profitability
        if 'interest_coverage' in fund_q.columns:
            ic = fund_q.loc[common, 'interest_coverage'].dropna()
            if len(ic) > 0:
                wp = w[ic.index]
                wp = wp / wp.sum()
                exposures['factor_profitability_ic'] = float((wp * ic).sum())
        
        if len(exposures) > 2:
            factor_results.append(exposures)

factor_df = pd.DataFrame(factor_results)
factor_df.to_parquet('quarterly_factor_exposures.parquet', index=False)
print(f"Saved {len(factor_df)} rows to quarterly_factor_exposures.parquet")

# ============================================================
# SUMMARY
# ============================================================
print("\n=== SUMMARY ===")
print(f"Look-through fundamentals: {len(lt_df)} rows, {len(available_metrics)} metrics")
print(f"Portfolio analytics: {len(price_df)} rows")
print(f"Concentration metrics: {len(conc_df)} rows")
print(f"Factor exposures: {len(factor_df)} rows")

# Show samples
print("\nLook-through sample:")
print(lt_df[['filer_ticker', 'as_of_date', 'lt_ev_ebitda', 'lt_roic', 'lt_fcf_margin', 'lt_debt_to_equity']].head(10).to_string())

print("\nPortfolio analytics sample:")
print(price_df[['filer_ticker', 'as_of_date', 'cum_return', 'vol_annualized', 'sharpe', 'max_drawdown', 'beta_spy']].head(10).to_string())

print("\nConcentration sample:")
print(conc_df[['filer_ticker', 'as_of_date', 'hhi', 'n_eff', 'top5_concentration', 'entropy_normalized', 'gini']].head(10).to_string())

print("\nFactor exposures sample:")
print(factor_df.head(10).to_string())